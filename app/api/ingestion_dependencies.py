import logging
from pathlib import Path

from app.services.ingestion_service import IngestionService
from app.services.indexing_service import IndexingService
from app.preprocessing.text_cleaner import TextCleaner
from app.preprocessing.chunker import TextChunker
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.vector_store import VectorStore
from app.services.document_registry import DocumentRegistry

logger = logging.getLogger(__name__)


class IngestionDependencies:
    """
    Creates components required for document ingestion.
    """

    def __init__(self, vector_store: VectorStore):
        logger.info("Loading ingestion components...")

        self.raw_directory = Path("data/raw")
        self.raw_directory.mkdir(parents=True, exist_ok=True)

        self.ingestion_service = IngestionService()
        self.text_cleaner = TextCleaner()

        self.text_chunker = TextChunker(
            chunk_size=1800,
            chunk_overlap=250
        )

        self.embedding_service = EmbeddingService()
        self.vector_store = vector_store
        self.document_registry = DocumentRegistry()

        self.indexing_service = IndexingService(
            text_cleaner=self.text_cleaner,
            text_chunker=self.text_chunker,
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            document_registry=self.document_registry
        )

        logger.info("Ingestion components loaded successfully.")