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

router = APIRouter()


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
}


@router.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@router.post(
    "/ask",
    response_model=AnswerResponse
)
def ask_question(
    request: Request,
    question_request: QuestionRequest
):
    """
    Answer a question using the RAG pipeline.
    """

    rag_pipeline = request.app.state.rag

    result = rag_pipeline.answer(
        question=question_request.question,
        retrieval_k=10,
        final_k=5
    )

    return result


@router.post("/ingest")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Upload and index a PDF or image document.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is missing."
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension}. "
                f"Supported types: PDF, JPG, JPEG, PNG."
            )
        )

    ingestion = request.app.state.ingestion

    destination = (
        ingestion.raw_directory /
        Path(file.filename).name
    )

    try:
        contents = await file.read()

        with destination.open(
            "wb"
        ) as output_file:

            output_file.write(contents)

        # Parse the uploaded document
        parser = ParserFactory.create(
            destination
        )

        document = parser.parse(
            destination
        )

        # Index the parsed document
        chunks_indexed = (
            ingestion.indexing_service.index_document(
                document
            )
        )

        return {
            "status": "success",
            "file_name": file.filename,
            "document_id": document.document_id,
            "chunks_indexed": chunks_indexed,
            "message": (
                "Document processed successfully."
            )
        }

    except Exception as exc:

        # Remove the file if processing failed
        if destination.exists():
            destination.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {exc}"
        )