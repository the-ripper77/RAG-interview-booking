"""Text extraction from uploaded files."""
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from src.utils.helpers import normalize_whitespace

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})


class UnsupportedFileError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(BytesIO(data))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".txt":
        text = data.decode("utf-8", errors="replace")
    else:
        raise UnsupportedFileError(f"Unsupported file type '{ext}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}")
    return normalize_whitespace(text)
