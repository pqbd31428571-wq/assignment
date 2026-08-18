from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker
from app.llm.llm_service import GroqLLM
from app.rag.prompt import PromptBuilder
from app.rag.rag_pipeline import RAGPipeline


class RAGDependencies:
    """
    Creates components required for the RAG pipeline.
    """

    def __init__(
        self,
        vector_store: VectorStore
    ):
        print("Loading RAG components...")

        self.embedding_service = EmbeddingService()

        self.vector_store = vector_store

        self.retriever = Retriever(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store
        )

        self.reranker = Reranker()

        self.llm = GroqLLM()

        self.prompt_builder = PromptBuilder()

        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            reranker=self.reranker,
            llm=self.llm,
            prompt_builder=self.prompt_builder
        )

        print("RAG components loaded successfully.")