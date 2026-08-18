from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker
from app.llm.llm_service import GroqLLM
from app.rag.prompt import PromptBuilder


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
        """
        Generate an answer using the RAG pipeline.

        Args:
            question: User's question.
            retrieval_k: Number of candidates retrieved.
            final_k: Number of chunks passed to the LLM.

        Returns:
            Answer and source information.
        """

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        # Step 1: Retrieve candidates
        retrieved_results = self.retriever.retrieve(
            query=question,
            top_k=retrieval_k
        )

        # Step 2: Rerank candidates
        reranked_results = self.reranker.rerank(
            query=question,
            results=retrieved_results,
            top_k=final_k
        )

        # Step 3: Build grounded prompt
        prompt = self.prompt_builder.build(
            question=question,
            contexts=reranked_results
        )

        # Step 4: Generate answer
        answer = self.llm.generate(prompt)

        # Step 5: Extract sources
        sources = []

        for result in reranked_results:

            payload = result.payload

            sources.append({
                "file_name": payload.get(
                    "file_name",
                    "Unknown"
                ),
                "chunk_id": payload.get(
                    "chunk_id",
                    "Unknown"
                ),
                "score": float(
                    result.score
                ),
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }