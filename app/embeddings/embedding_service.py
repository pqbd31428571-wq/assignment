from typing import List

from sentence_transformers import SentenceTransformer

from app.preprocessing.chunker import DocumentChunk


class EmbeddingService:
    """
    Service responsible for converting text into embeddings.
    """

    MODEL_NAME = "BAAI/bge-base-en-v1.5"

    def __init__(self):
        """
        Load the embedding model.
        """

        self.model = SentenceTransformer(
            self.MODEL_NAME
        )

    def embed_text(self, text: str):
        """
        Convert a single piece of text into an embedding.

        Args:
            text: Text to embed.

        Returns:
            Numerical embedding vector.
        """

        if not text.strip():
            raise ValueError(
                "Cannot generate embedding for empty text."
            )

        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    def embed_chunks(
        self,
        chunks: List[DocumentChunk]
    ):
        """
        Generate embeddings for multiple document chunks.

        Args:
            chunks: List of DocumentChunk objects.

        Returns:
            List of embedding vectors.
        """

        if not chunks:
            return []

        texts = [
            chunk.content
            for chunk in chunks
        ]

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )