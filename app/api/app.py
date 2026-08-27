import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
torch._dynamo.config.disable = True

from app.config.logging_config import setup_logging
setup_logging()

import logging
import time

from fastapi import FastAPI, Request

from app.api.routes import router
from app.rag.dependencies import RAGDependencies
from app.api.ingestion_dependencies import IngestionDependencies
from app.vectorstore.vector_store import VectorStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document RAG API",
    description="RAG-based document question answering system",
    version="1.0.0"
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Logs every request with its path, status code, and duration — so
    you can see in logs/app.log whether /ask or /ingest-directory is
    slow, instead of guessing or watching a spinner.
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


logger.info("Creating VectorStore...")
vector_store = VectorStore()

logger.info("Loading RAG dependencies...")
rag_dependencies = RAGDependencies(vector_store=vector_store)

logger.info("Loading ingestion dependencies...")
ingestion_dependencies = IngestionDependencies(vector_store=vector_store)

app.state.rag = rag_dependencies.pipeline
app.state.ingestion = ingestion_dependencies

app.include_router(router)

logger.info("Application startup complete.")


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down. Closing VectorStore.")
    vector_store.close()