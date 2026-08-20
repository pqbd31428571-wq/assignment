from pathlib import Path
from uuid import uuid5, NAMESPACE_DNS

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from app.preprocessing.chunker import DocumentChunk


class VectorStore:
    """
    Handles storage and retrieval of document embeddings
    using Qdrant.
    """

    COLLECTION_NAME = "documents"

    VECTOR_SIZE = 768

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
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings
    ):
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        if not chunks:
            return

        points = []

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):
            point_id = str(
                uuid5(
                    NAMESPACE_DNS,
                    chunk.chunk_id
                )
            )

            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
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
        query_embedding,
        limit: int = 5
    ):
        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=limit,
            with_payload=True,
        )

        return results.points

    def close(self):
        self.client.close()