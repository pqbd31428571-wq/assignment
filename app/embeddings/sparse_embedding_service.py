import logging
from typing import List

from fastembed import SparseTextEmbedding

from app.preprocessing.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class SparseEmbeddingService:
    """
    Service responsible for converting text into BM25 sparse
    vectors, for keyword-based retrieval alongside dense semantic
    search.

    Uses fastembed's "Qdrant/bm25" model, which produces sparse
    (indices, values) vectors without needing to fit BM25 over the
    whole corpus up front. Document-side vectors carry raw
    term-frequency saturation; Qdrant applies IDF weighting
    server-side at query time (see Modifier.IDF in VectorStore), so
    the IDF statistics stay correct as new documents are indexed —
    no periodic refit required.

    Note the model distinguishes document embedding (`embed`) from
    query embedding (`query_embed`): document vectors encode term
    frequency, query vectors just mark which terms are present.
    Mixing them up silently produces poor retrieval quality.
    """

    MODEL_NAME = "Qdrant/bm25"

    def __init__(self):
        logger.info("Loading sparse embedding model '%s'...", self.MODEL_NAME)
        self.model = SparseTextEmbedding(model_name=self.MODEL_NAME)
        logger.info("Sparse embedding model ready.")

    def embed_text(self, text: str):
        """
        Embed a single query string. Use this for user questions,
        not for document chunks (see embed_chunks).
        """

        if not text.strip():
            raise ValueError("Cannot generate a sparse embedding for empty text.")

        return next(self.model.query_embed(text))

    def embed_chunks(self, chunks: List[DocumentChunk]):
        """
        Embed a batch of document chunks for indexing.
        """

        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]

        logger.info("Sparse-embedding %d chunk(s)...", len(texts))

        embeddings = list(self.model.embed(texts))

        logger.info("Finished sparse-embedding %d chunk(s).", len(texts))

        return embeddings
