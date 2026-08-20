from typing import List

import torch
from sentence_transformers import SentenceTransformer

from app.preprocessing.chunker import DocumentChunk


class EmbeddingService:
    """
    Service responsible for converting text into embeddings.
    """

    MODEL_NAME = "BAAI/bge-base-en-v1.5"

    def __init__(self, batch_size: int = 64):
        """
        Load the embedding model. Uses a GPU automatically if one is
        available; falls back to CPU otherwise (same behavior you
        have now, just faster if a GPU ever becomes available).
        """

        self.batch_size = batch_size
        self.device = self._resolve_device()

        self.model = SentenceTransformer(
            self.MODEL_NAME,
            device=self.device
        )

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def embed_text(self, text: str):
        if not text.strip():
            raise ValueError(
                "Cannot generate embedding for empty text."
            )

        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    def embed_chunks(self, chunks: List[DocumentChunk]):
        """
        Generate embeddings for multiple document chunks in batches —
        without batch_size, sentence-transformers uses its own
        default, which is less efficient when embedding hundreds of
        chunks at once during bulk ingestion.
        """

        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]

        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )