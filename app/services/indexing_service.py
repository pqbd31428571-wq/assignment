from pathlib import Path
from typing import List, Tuple

from app.models.document import Document
from app.preprocessing.text_cleaner import TextCleaner
from app.preprocessing.chunker import DocumentChunk, TextChunker
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.vector_store import VectorStore
from app.services.document_registry import DocumentRegistry


class IndexingService:
    """
    Coordinates the document indexing pipeline.
    """

    def __init__(
        self,
        text_cleaner: TextCleaner,
        text_chunker: TextChunker,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        document_registry: DocumentRegistry
    ):
        self.text_cleaner = text_cleaner
        self.text_chunker = text_chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.document_registry = document_registry

    def index_document(self, document: Document) -> int:
        """
        Process and index a single document. Used by the /ingest
        upload route.
        """
        return self._index_batch([document])

    def index_documents(self, documents: List[Document]) -> int:
        """
        Index multiple documents.

        PERFORMANCE NOTE: the original implementation called
        index_document() in a loop, issuing one embed_chunks() call
        per document — for ~100 documents that's ~100 separate model
        calls. This version still skip-checks and registers each
        document individually (preserving the dedup behavior), but
        embeds every chunk from every non-skipped document in ONE
        batched call before writing to the vector store.
        """
        return self._index_batch(documents)

    def _index_batch(self, documents: List[Document]) -> int:
        pending_chunks: List[DocumentChunk] = []
        pending_docs: List[Tuple[Document, Path, int]] = []

        for document in documents:
            source_path = Path(document.source_path)

            if self.document_registry.is_indexed(source_path):
                print(f"Skipping already indexed: {document.file_name}")
                continue

            document.content = self.text_cleaner.clean(document.content)
            chunks = self.text_chunker.chunk_document(document)

            if not chunks:
                print(f"No usable content found: {document.file_name}")
                continue

            pending_chunks.extend(chunks)
            pending_docs.append((document, source_path, len(chunks)))

        if not pending_chunks:
            return 0

        # One batched embedding call across every pending chunk from
        # every pending document, instead of one call per document.
        embeddings = self.embedding_service.embed_chunks(pending_chunks)
        self.vector_store.add_chunks(pending_chunks, embeddings)

        total_indexed = 0

        # Register each document only after its vectors are confirmed
        # stored — same "register on success only" guarantee as before.
        for document, source_path, count in pending_docs:
            self.document_registry.register(source_path, document.document_id)
            print(f"Indexed {count} chunks from {document.file_name}")
            total_indexed += count

        return total_indexed