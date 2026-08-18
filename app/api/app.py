from fastapi import FastAPI

from app.api.routes import router
from app.rag.dependencies import RAGDependencies
from app.api.ingestion_dependencies import IngestionDependencies
from app.vectorstore.vector_store import VectorStore


app = FastAPI(
    title="Document RAG API",
    description="RAG-based document question answering system",
    version="1.0.0"
)


# Create ONE Qdrant instance
vector_store = VectorStore()


# Give the same VectorStore to both services
rag_dependencies = RAGDependencies(
    vector_store=vector_store
)

ingestion_dependencies = IngestionDependencies(
    vector_store=vector_store
)


app.state.rag = rag_dependencies.pipeline
app.state.ingestion = ingestion_dependencies


app.include_router(router)


@app.on_event("shutdown")
def shutdown_event():
    vector_store.close()