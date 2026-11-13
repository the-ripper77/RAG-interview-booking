"""Custom retrieval: embed the query, similarity search in Milvus."""
from src.config import get_settings
from src.embeddings.embedder import embed_query
from src.vectordb.vector_store import ChunkHit, search


def retrieve(query: str, top_k: int | None = None) -> list[ChunkHit]:
    return search(embed_query(query), top_k or get_settings().retrieval.top_k)


def format_context(hits: list[ChunkHit]) -> str:
    if not hits:
        return "No relevant passages found in the knowledge base."
    return "\n\n".join(
        f'<passage source="{h["filename"]}" chunk="{h["chunk_index"]}" score="{h["score"]:.3f}">\n'
        f'{h["text"]}\n</passage>'
        for h in hits
    )
