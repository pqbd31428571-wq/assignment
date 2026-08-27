import logging
from typing import List

import torch
from sentence_transformers import SentenceTransformer

from app.preprocessing.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service responsible for converting text into embeddings.
    """

    MODEL_NAME = "BAAI/bge-base-en-v1.5"

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self.device = self._resolve_device()

        logger.info(
            "Loading embedding model '%s' on device '%s'...",
            self.MODEL_NAME, self.device,
        )
        self.model = SentenceTransformer(self.MODEL_NAME, device=self.device)
        logger.info("Embedding model ready.")

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def embed_text(self, text: str):
        if not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")

        return self.model.encode(text, normalize_embeddings=True)

    def embed_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]

        logger.info("Embedding %d chunk(s)...", len(texts))

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        logger.info("Finished embedding %d chunk(s).", len(texts))

        return embeddings