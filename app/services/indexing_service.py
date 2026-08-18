from pathlib import Path
from app.models.document import Document
from app.preprocessing.text_cleaner import TextCleaner
from app.preprocessing.chunker import TextChunker
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

    def index_document(
        self,
        document: Document
    ) -> int:
        """
        Process and index a single document.

        Returns:
            Number of indexed chunks.
        """

        source_path = Path(document.source_path)

        # Skip already indexed documents
        if self.document_registry.is_indexed(
            source_path
        ):
            print(
                f"Skipping already indexed: "
                f"{document.file_name}"
            )

            return 0

        # 1. Clean text
        document.content = self.text_cleaner.clean(
            document.content
        )

        # 2. Create chunks
        chunks = self.text_chunker.chunk_document(
            document
        )

        if not chunks:
            print(
                f"No usable content found: "
                f"{document.file_name}"
            )

            return 0

        # 3. Generate embeddings in batch
        embeddings = self.embedding_service.embed_chunks(
            chunks
        )

        # 4. Store vectors
        self.vector_store.add_chunks(
            chunks,
            embeddings
        )

        # 5. Register document only after
        # successful indexing
        self.document_registry.register(
            source_path,
            document.document_id
        )

        print(
            f"Indexed {len(chunks)} chunks "
            f"from {document.file_name}"
        )

        return len(chunks)

    def index_documents(
        self,
        documents: list[Document]
    ) -> int:
        """
        Index multiple documents.
        """

        total_chunks = 0

        for document in documents:

            total_chunks += self.index_document(
                document
            )

        return total_chunks