import logging
import time

from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker
from app.llm.llm_service import GroqLLM
from app.rag.prompt import PromptBuilder

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Coordinates retrieval, reranking, prompt construction,
    and LLM generation.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        llm: GroqLLM,
        prompt_builder: PromptBuilder
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.prompt_builder = prompt_builder

    def answer(
        self,
        question: str,
        retrieval_k: int = 10,
        final_k: int = 5
    ) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        start = time.perf_counter()

        retrieved_results = self.retriever.retrieve(
            query=question,
            top_k=retrieval_k
        )
        t_retrieve = time.perf_counter()

        reranked_results = self.reranker.rerank(
            query=question,
            results=retrieved_results,
            top_k=final_k
        )
        t_rerank = time.perf_counter()

        prompt = self.prompt_builder.build(
            question=question,
            contexts=reranked_results
        )

        answer = self.llm.generate(prompt)
        t_generate = time.perf_counter()

        logger.info(
            "Timings for '%s' -> retrieve: %.0fms, rerank: %.0fms, "
            "generate: %.0fms, total: %.0fms",
            question,
            (t_retrieve - start) * 1000,
            (t_rerank - t_retrieve) * 1000,
            (t_generate - t_rerank) * 1000,
            (t_generate - start) * 1000,
        )

        sources = []

        for result in reranked_results:
            payload = result.payload

            sources.append({
                "file_name": payload.get("file_name", "Unknown"),
                "chunk_id": payload.get("chunk_id", "Unknown"),
                "score": float(result.score),
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }