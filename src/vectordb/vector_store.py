"""Milvus / Zilliz Cloud vector store."""
from functools import lru_cache
from typing import TypedDict

from pymilvus import MilvusClient

from src.config import get_settings


class ChunkHit(TypedDict):
    text: str
    document_id: str
    filename: str
    chunk_index: int
    score: float


_OUTPUT_FIELDS = ["text", "document_id", "filename", "chunk_index"]


@lru_cache
def get_client() -> MilvusClient:
    s = get_settings()
    client = MilvusClient(uri=s.milvus_uri, token=s.milvus_token)
    if not client.has_collection(s.vectordb.collection):
        # Quick-setup collection: auto-id int64 pk + vector + dynamic fields for metadata.
        client.create_collection(
            collection_name=s.vectordb.collection,
            dimension=s.embeddings.dimension,
            metric_type="COSINE",
            auto_id=True,
        )
    return client


def insert_chunks(document_id: str, filename: str, chunks: list[str], vectors: list[list[float]]) -> None:
    rows = [
        {"vector": v, "text": t, "document_id": document_id, "filename": filename, "chunk_index": i}
        for i, (t, v) in enumerate(zip(chunks, vectors, strict=True))
    ]
    get_client().insert(collection_name=get_settings().vectordb.collection, data=rows)


def search(vector: list[float], top_k: int) -> list[ChunkHit]:
    results = get_client().search(
        collection_name=get_settings().vectordb.collection,
        data=[vector],
        limit=top_k,
        output_fields=_OUTPUT_FIELDS,
    )
    return [ChunkHit(**{f: hit["entity"][f] for f in _OUTPUT_FIELDS}, score=hit["distance"]) for hit in results[0]]
