import torch
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranks retrieved document chunks according to
    their relevance to the user's query.
    """

    MODEL_NAME = "BAAI/bge-reranker-base"

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.device = self._resolve_device()

        self.model = CrossEncoder(
            self.MODEL_NAME,
            device=self.device
        )

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def rerank(
        self,
        query: str,
        results,
        top_k: int = 5
    ):
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not results:
            return []

        documents = [
            result.payload.get("content", "")
            for result in results
        ]

        pairs = [
            [query, document]
            for document in documents
        ]

        scores = self.model.predict(pairs, batch_size=self.batch_size)

        scored_results = list(
            zip(results, scores)
        )

        scored_results.sort(
            key=lambda item: float(item[1]),
            reverse=True
        )

        return [
            result
            for result, score in scored_results[:top_k]
        ]