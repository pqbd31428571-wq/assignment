from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranks retrieved document chunks according to
    their relevance to the user's query.
    """

    MODEL_NAME = "BAAI/bge-reranker-base"

    def __init__(self):
        """
        Load the reranking model.
        """

        self.model = CrossEncoder(
            self.MODEL_NAME
        )

    def rerank(
        self,
        query: str,
        results,
        top_k: int = 5
    ):
        """
        Rerank retrieved results.

        Args:
            query: User's question.
            results: Results returned by the vector database.
            top_k: Number of final results to return.

        Returns:
            Reranked results.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not results:
            return []

        # Extract the text from Qdrant payloads
        documents = [
            result.payload.get("content", "")
            for result in results
        ]

        # Create query-document pairs
        pairs = [
            [query, document]
            for document in documents
        ]

        # Calculate relevance scores
        scores = self.model.predict(pairs)

        # Combine results with their scores
        scored_results = list(
            zip(results, scores)
        )

        # Sort from most relevant to least relevant
        scored_results.sort(
            key=lambda item: float(item[1]),
            reverse=True
        )

        # Return only the best results
        return [
            result
            for result, score in scored_results[:top_k]
        ]