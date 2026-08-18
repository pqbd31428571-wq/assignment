from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.vector_store import VectorStore


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
        """
        Initialize the retriever.

        Args:
            embedding_service: Service used to create query embeddings.
            vector_store: Vector database used for similarity search.
        """

        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: User's question.
            top_k: Number of relevant chunks to retrieve.

        Returns:
            List of relevant search results.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        # Convert the user's question into an embedding
        query_embedding = self.embedding_service.embed_text(
            query
        )

        # Search the vector database
        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=top_k
        )

        return results