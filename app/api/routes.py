import logging
from pathlib import Path

from fastapi import (
    APIRouter,
    Request,
    UploadFile,
    File,
    HTTPException,
)

from app.api.schemas import (
    QuestionRequest,
    AnswerResponse,
)
from app.parsers.parser_factory import ParserFactory

logger = logging.getLogger(__name__)

router = APIRouter()


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
}


@router.get("/health")
def health_check():
    return {"status": "healthy"}


@router.post("/ask", response_model=AnswerResponse)
def ask_question(request: Request, question_request: QuestionRequest):
    logger.info("Question received: %s", question_request.question)

    rag_pipeline = request.app.state.rag

    try:
        result = rag_pipeline.answer(
            question=question_request.question,
            retrieval_k=6,
            final_k=5
        )
    except Exception:
        logger.exception("Failed to answer question: %s", question_request.question)
        raise HTTPException(status_code=500, detail="Failed to generate an answer.")

    logger.info(
        "Answered with %d source(s) for question: %s",
        len(result["sources"]), question_request.question,
    )

    return result


@router.post("/ingest")
async def ingest_document(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is missing.")

    extension = Path(file.filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension}. "
                f"Supported types: PDF, JPG, JPEG, PNG."
            )
        )

    ingestion = request.app.state.ingestion
    destination = ingestion.raw_directory / Path(file.filename).name

    try:
        contents = await file.read()

        with destination.open("wb") as output_file:
            output_file.write(contents)

        parser = ParserFactory.create(destination)
        document = parser.parse(destination)

        chunks_indexed = ingestion.indexing_service.index_document(document)

        logger.info(
            "Uploaded and indexed %s: %d chunk(s)",
            file.filename, chunks_indexed,
        )

        return {
            "status": "success",
            "file_name": file.filename,
            "document_id": document.document_id,
            "chunks_indexed": chunks_indexed,
            "message": "Document processed successfully."
        }

    except Exception as exc:
        logger.exception("Failed to process uploaded file: %s", file.filename)

        if destination.exists():
            destination.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {exc}"
        )


@router.post("/ingest-directory")
def ingest_directory(request: Request):
    logger.info("Bulk ingestion triggered.")

    ingestion = request.app.state.ingestion

    documents = ingestion.ingestion_service.ingest_directory(
        ingestion.raw_directory,
        skip_if=ingestion.document_registry.is_indexed,
    )

    if not documents:
        logger.info("Bulk ingestion found nothing new to process.")
        return {
            "status": "success",
            "documents_found": 0,
            "chunks_indexed": 0,
            "message": (
                "No new supported files found in data/raw/ "
                "(everything already indexed, or the folder is empty)."
            ),
        }

    chunks_indexed = ingestion.indexing_service.index_documents(documents)

    logger.info(
        "Bulk ingestion complete: %d document(s), %d chunk(s) indexed.",
        len(documents), chunks_indexed,
    )

    return {
        "status": "success",
        "documents_found": len(documents),
        "chunks_indexed": chunks_indexed,
        "message": "Bulk ingestion complete.",
    }