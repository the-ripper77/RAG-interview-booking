"""Local embeddings via fastembed (ONNX, no torch, no API key)."""
from functools import lru_cache

from fastembed import TextEmbedding

from src.config import get_settings


@lru_cache
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=get_settings().embeddings.model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _model().embed(texts)]


def embed_query(text: str) -> list[float]:
    return next(iter(_model().query_embed(text))).tolist()
