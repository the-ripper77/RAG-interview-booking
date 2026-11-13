import datetime as dt
import uuid

from pydantic import BaseModel, ConfigDict, Field

from src.chunking.chunker import ChunkingStrategy


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_type: str
    size_bytes: int
    chunking_strategy: ChunkingStrategy
    num_chunks: int
    created_at: dt.datetime


class BookingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    name: str
    email: str
    date: dt.date
    time: dt.time
    created_at: dt.datetime


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=100)
    message: str = Field(min_length=1, max_length=4000)


class Source(BaseModel):
    filename: str
    chunk_index: int
    score: float
    text: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source]
    booking: BookingOut | None = None
