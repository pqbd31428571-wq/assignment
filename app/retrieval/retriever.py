import logging

from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves the most relevant document chunks
    for a user's question.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.embedding_service.embed_text(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=top_k
        )

        logger.info("Retrieved %d candidate(s) for query: %s", len(results), query)

        return results