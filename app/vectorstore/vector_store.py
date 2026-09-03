from pathlib import Path
from uuid import uuid5, NAMESPACE_DNS

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    Modifier,
    PointStruct,
    Prefetch,
    FusionQuery,
    Fusion,
)

from app.preprocessing.chunker import DocumentChunk


class VectorStore:
    """
    Handles storage and retrieval of document embeddings
    using Qdrant.

    Each point stores two named vectors:
      - "dense":  semantic embedding (BAAI/bge-base-en-v1.5)
      - "sparse": BM25 term vector (fastembed's Qdrant/bm25 model),
                   with Qdrant computing IDF statistics server-side
                   (Modifier.IDF) so they stay correct as the
                   collection grows — no manual refitting.

    search() runs both retrievals in one request and fuses them
    with Reciprocal Rank Fusion (RRF), so results benefit from both
    semantic similarity and exact keyword/term matches (IDs,
    acronyms, rare technical terms) that dense embeddings alone
    tend to miss.

    NOTE — SCHEMA CHANGE: this collection now uses named vectors
    instead of a single unnamed vector. It is NOT compatible with a
    collection created by the previous version of this file. Delete
    your existing `vector_db/` directory and re-run ingestion
    (/ingest-directory) after upgrading, or point-lookups and
    upserts will fail against the old schema.
    """

    COLLECTION_NAME = "documents"

    VECTOR_SIZE = 768

    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    # Upsert in pages so bulk-ingesting ~100 documents' worth of
    # chunks (via /ingest-directory) never sends one oversized
    # request to Qdrant in a single call.
    UPSERT_BATCH_SIZE = 256

    def __init__(
        self,
        storage_path: str = "vector_db"
    ):
        Path(storage_path).mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = QdrantClient(
            path=storage_path
        )

        self._create_collection()

    def _create_collection(self):
        collections = self.client.get_collections()

        collection_names = [
            collection.name
            for collection in collections.collections
        ]

        if self.COLLECTION_NAME not in collection_names:

            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config={
                    self.DENSE_VECTOR_NAME: VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: SparseVectorParams(
                        modifier=Modifier.IDF,
                    ),
                },
            )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        dense_embeddings,
        sparse_embeddings,
    ):
        if (
            len(chunks) != len(dense_embeddings)
            or len(chunks) != len(sparse_embeddings)
        ):
            raise ValueError(
                "Number of chunks, dense embeddings, and sparse "
                "embeddings must all match."
            )

        if not chunks:
            return

        points = []

        for chunk, dense_embedding, sparse_embedding in zip(
            chunks,
            dense_embeddings,
            sparse_embeddings,
        ):
            point_id = str(
                uuid5(
                    NAMESPACE_DNS,
                    chunk.chunk_id
                )
            )

            point = PointStruct(
                id=point_id,
                vector={
                    self.DENSE_VECTOR_NAME: dense_embedding.tolist(),
                    self.SPARSE_VECTOR_NAME: SparseVector(
                        indices=sparse_embedding.indices.tolist(),
                        values=sparse_embedding.values.tolist(),
                    ),
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    **chunk.metadata,
                },
            )

            points.append(point)

        for start in range(0, len(points), self.UPSERT_BATCH_SIZE):
            batch = points[start:start + self.UPSERT_BATCH_SIZE]
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=batch,
            )

    def search(
        self,
        dense_query_embedding,
        sparse_query_embedding,
        limit: int = 5,
        prefetch_limit: int = 20,
    ):
        """
        Hybrid search: retrieves `prefetch_limit` candidates from
        the dense vector and `prefetch_limit` from the sparse
        vector independently, then fuses the two ranked lists with
        Reciprocal Rank Fusion into a single ranking of `limit`
        results.

        Keep prefetch_limit comfortably larger than limit (roughly
        3-4x) so RRF has enough candidates from each side to work
        with — too small a prefetch defeats the point of fusing two
        retrieval methods.

        NOTE: the returned `result.score` is now an RRF fusion
        score, not a cosine similarity — it's a different scale
        than before (no longer bounded to [0, 1]). If you display
        this score to users (e.g. Streamlit source list), you'll
        want to relabel it or switch to displaying the reranker's
        score instead, since RRF scores aren't intuitively
        interpretable as "relevance percentage" the way cosine
        similarity was.
        """

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=dense_query_embedding.tolist(),
                    using=self.DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query_embedding.indices.tolist(),
                        values=sparse_query_embedding.values.tolist(),
                    ),
                    using=self.SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        return results.points

    def close(self):
        self.client.close()
