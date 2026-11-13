import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.schemas import BookingOut, ChatRequest, ChatResponse, DocumentOut, Source
from src.chunking.chunker import ChunkingStrategy, chunk_text
from src.config import get_settings
from src.db.models import Booking, Document, get_db
from src.embeddings.embedder import embed_documents
from src.ingestion.loader import UnsupportedFileError, extract_text
from src.llm.llm_client import chat
from src.vectordb.vector_store import insert_chunks

log = logging.getLogger(__name__)
router = APIRouter()
DB = Annotated[Session, Depends(get_db)]


@router.post("/documents", response_model=DocumentOut, status_code=status.HTTP_201_CREATED, tags=["ingestion"])
def upload_document(
    file: UploadFile,
    db: DB,
    chunking_strategy: Annotated[ChunkingStrategy, Form()] = ChunkingStrategy.SENTENCE,
) -> Document:
    """Upload a .pdf or .txt file: extract text, chunk, embed, store vectors in Milvus and metadata in SQL."""
    s = get_settings()
    data = file.file.read()
    if len(data) > s.ingestion.max_file_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"File exceeds {s.ingestion.max_file_mb} MB")

    filename = file.filename or "upload"
    try:
        text = extract_text(filename, data)
    except UnsupportedFileError as e:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(e)) from e
    if not text:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "No extractable text (scanned PDF?)")

    chunks = chunk_text(text, chunking_strategy, s.chunking.chunk_size, s.chunking.chunk_overlap)
    doc_id = str(uuid.uuid4())
    insert_chunks(doc_id, filename, chunks, embed_documents(chunks))

    # Metadata is written only after vectors are stored, so the SQL table never lists a document
    # that retrieval can't find.
    doc = Document(
        id=doc_id,
        filename=filename,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=len(data),
        chunking_strategy=chunking_strategy.value,
        num_chunks=len(chunks),
    )
    db.add(doc)
    db.commit()
    log.info("ingested %s: %d chunks (%s)", filename, len(chunks), chunking_strategy)
    return doc


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat_endpoint(req: ChatRequest, db: DB) -> ChatResponse:
    """Multi-turn RAG chat. Reuse `session_id` across turns; omit it to start a new conversation."""
    result = chat(req.session_id, req.message, db)
    return ChatResponse(
        session_id=req.session_id,
        answer=result.answer,
        sources=[Source.model_validate(h) for h in result.sources],
        booking=BookingOut.model_validate(result.booking) if result.booking else None,
    )


@router.get("/bookings", response_model=list[BookingOut], tags=["chat"])
def list_bookings(db: DB) -> list[Booking]:
    return list(db.scalars(select(Booking).order_by(Booking.date, Booking.time)))
