import logging
from pathlib import Path
from typing import List, Tuple

from app.models.document import Document
from app.preprocessing.text_cleaner import TextCleaner
from app.preprocessing.chunker import DocumentChunk, TextChunker
from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.sparse_embedding_service import SparseEmbeddingService
from app.vectorstore.vector_store import VectorStore
from app.services.document_registry import DocumentRegistry

logger = logging.getLogger(__name__)


class IndexingService:
    """
    Coordinates the document indexing pipeline.
    """

    def __init__(
        self,
        text_cleaner: TextCleaner,
        text_chunker: TextChunker,
        embedding_service: EmbeddingService,
        sparse_embedding_service: SparseEmbeddingService,
        vector_store: VectorStore,
        document_registry: DocumentRegistry
    ):
        self.text_cleaner = text_cleaner
        self.text_chunker = text_chunker
        self.embedding_service = embedding_service
        self.sparse_embedding_service = sparse_embedding_service
        self.vector_store = vector_store
        self.document_registry = document_registry

    def index_document(self, document: Document) -> int:
        return self._index_batch([document])

    def index_documents(self, documents: List[Document]) -> int:
        return self._index_batch(documents)

    def _index_batch(self, documents: List[Document]) -> int:
        pending_chunks: List[DocumentChunk] = []
        pending_docs: List[Tuple[Document, Path, int]] = []

        for document in documents:
            source_path = Path(document.source_path)

            if self.document_registry.is_indexed(source_path):
                logger.info("Skipping already indexed: %s", document.file_name)
                continue

            document.content = self.text_cleaner.clean(document.content)
            chunks = self.text_chunker.chunk_document(document)

            if not chunks:
                logger.warning("No usable content found: %s", document.file_name)
                continue

            pending_chunks.extend(chunks)
            pending_docs.append((document, source_path, len(chunks)))

        if not pending_chunks:
            logger.info("Nothing new to index in this batch.")
            return 0

        logger.info(
            "Embedding %d chunk(s) from %d document(s)...",
            len(pending_chunks), len(pending_docs),
        )
        dense_embeddings = self.embedding_service.embed_chunks(pending_chunks)
        sparse_embeddings = self.sparse_embedding_service.embed_chunks(pending_chunks)

        self.vector_store.add_chunks(
            pending_chunks,
            dense_embeddings,
            sparse_embeddings,
        )

        total_indexed = 0

        for document, source_path, count in pending_docs:
            self.document_registry.register(source_path, document.document_id)
            logger.info("Indexed %d chunk(s) from %s", count, document.file_name)
            total_indexed += count

        return total_indexed
