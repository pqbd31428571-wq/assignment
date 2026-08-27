import logging

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranks retrieved document chunks according to
    their relevance to the user's query.
    """

    MODEL_NAME = "BAAI/bge-reranker-base"

    def __init__(self, batch_size: int = 32, max_length: int = 384):
        self.batch_size = batch_size
        self.device = self._resolve_device()

        logger.info(
            "Loading reranker model '%s' on device '%s' (max_length=%d)...",
            self.MODEL_NAME, self.device, max_length,
        )

        # max_length caps how many tokens of each query+document pair
        # the model actually processes. Cross-encoder cost scales
        # roughly with sequence length, so this is the single biggest
        # lever available without changing models — your chunks can
        # run up to ~1800 characters (~400-450 tokens), and capping at
        # 384 means the model spends compute on the part of the chunk
        # that matters most (the beginning — titles/headers/labels)
        # instead of scoring every character. Lower this further
        # (e.g. 256) for more speed at some relevance cost.
        self.model = CrossEncoder(
            self.MODEL_NAME,
            device=self.device,
            max_length=max_length,
        )

        logger.info("Reranker ready.")

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def rerank(self, query: str, results, top_k: int = 5):
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not results:
            logger.info("No candidates to rerank.")
            return []

        documents = [result.payload.get("content", "") for result in results]
        pairs = [[query, document] for document in documents]

        scores = self.model.predict(pairs, batch_size=self.batch_size)

        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda item: float(item[1]), reverse=True)

        top_results = [result for result, score in scored_results[:top_k]]

        logger.info(
            "Reranked %d candidate(s) down to top %d.",
            len(results), len(top_results),
        )

        return top_results