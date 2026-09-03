import logging

from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.sparse_embedding_service import SparseEmbeddingService
from app.vectorstore.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves the most relevant document chunks for a user's
    question, using hybrid search (dense semantic + BM25 sparse,
    fused with RRF in VectorStore).
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        sparse_embedding_service: SparseEmbeddingService,
        vector_store: VectorStore
    ):
        self.embedding_service = embedding_service
        self.sparse_embedding_service = sparse_embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        dense_query_embedding = self.embedding_service.embed_text(query)
        sparse_query_embedding = self.sparse_embedding_service.embed_text(query)

        results = self.vector_store.search(
            dense_query_embedding=dense_query_embedding,
            sparse_query_embedding=sparse_query_embedding,
            limit=top_k
        )

        logger.info("Retrieved %d candidate(s) for query: %s", len(results), query)

        return results
