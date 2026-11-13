"""Two chunking strategies, selectable per upload.

- fixed:    sliding character window with overlap. Predictable sizes, may cut mid-sentence.
- sentence: packs whole sentences into chunks up to chunk_size. Keeps semantic units intact.
"""
import re
from enum import StrEnum


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    SENTENCE = "sentence"


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    return [c for i in range(0, len(text), step) if (c := text[i : i + chunk_size].strip())]


def sentence_chunks(text: str, chunk_size: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for sentence in (s.strip() for s in _SENTENCE_SPLIT.split(text)):
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > chunk_size:
            chunks.append(current)
            current = ""
        if len(sentence) > chunk_size:  # a single oversized sentence falls back to fixed windows
            chunks.extend(fixed_size_chunks(sentence, chunk_size, 0))
            continue
        current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, strategy: ChunkingStrategy, chunk_size: int, overlap: int) -> list[str]:
    if strategy is ChunkingStrategy.FIXED:
        return fixed_size_chunks(text, chunk_size, overlap)
    return sentence_chunks(text, chunk_size)
