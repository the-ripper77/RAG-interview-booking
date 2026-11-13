"""Offline tests: no Ollama, Redis or Milvus needed (those are faked)."""
import datetime as dt
import json
import os
import tempfile
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MILVUS_URI", "http://localhost:19530")
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/test.db"

import pytest

from src.booking.service import BookingRequest
from src.chunking.chunker import ChunkingStrategy, chunk_text, fixed_size_chunks, sentence_chunks
from src.db.models import Booking, SessionLocal, init_db
from src.ingestion.loader import UnsupportedFileError, extract_text
from src.llm import llm_client

init_db()
TOMORROW = (dt.date.today() + dt.timedelta(days=1)).isoformat()


# --- chunking ---------------------------------------------------------------

def test_fixed_chunks_overlap_and_coverage() -> None:
    text = "abcdefghij" * 10  # 100 chars
    chunks = fixed_size_chunks(text, chunk_size=30, overlap=10)
    assert all(len(c) <= 30 for c in chunks)
    assert chunks[0][-10:] == chunks[1][:10]  # overlap preserved
    assert chunks[-1].endswith(text[-5:])  # tail not lost


def test_sentence_chunks_keep_sentences_whole() -> None:
    text = "First sentence here. Second one is here! Third? Fourth sentence."
    chunks = sentence_chunks(text, chunk_size=45)
    assert all(len(c) <= 45 for c in chunks)
    assert " ".join(chunks) == text  # nothing dropped, no sentence split


def test_sentence_chunks_oversized_sentence_falls_back() -> None:
    chunks = sentence_chunks("x" * 120, chunk_size=50)
    assert [len(c) for c in chunks] == [50, 50, 20]


def test_chunk_text_dispatch_and_bad_overlap() -> None:
    assert chunk_text("a. b.", ChunkingStrategy.SENTENCE, 100, 10) == ["a. b."]
    with pytest.raises(ValueError):
        fixed_size_chunks("abc", chunk_size=10, overlap=10)


# --- loader -----------------------------------------------------------------

def test_extract_txt_and_reject_other_types() -> None:
    assert extract_text("a.txt", b"hello   world\n\n\n\nbye") == "hello world\n\nbye"
    with pytest.raises(UnsupportedFileError):
        extract_text("a.docx", b"...")


# --- booking validation -----------------------------------------------------

def test_booking_rejects_bad_email_and_past_date() -> None:
    with pytest.raises(ValueError):
        BookingRequest(name="Ann Lee", email="not-an-email", date=TOMORROW, time="10:00")
    with pytest.raises(ValueError):
        BookingRequest(name="Ann Lee", email="ann@example.com", date="2000-01-01", time="10:00")


# --- chat tool loop ---------------------------------------------------------

class FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list[Any] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in self.tool_calls or []
            ],
        }
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


class FakeCompletions:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def _resp(content: str | None = None, tool_calls: list[Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=FakeMessage(content, tool_calls))])


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    store: dict[str, list[Any]] = {}
    monkeypatch.setattr(llm_client.chat_memory, "load_history", lambda sid: list(store.get(sid, [])))
    monkeypatch.setattr(llm_client.chat_memory, "append_messages", lambda sid, m: store.setdefault(sid, []).extend(m))
    hit = {"text": "Python and SQL required.", "document_id": "d1", "filename": "jd.txt", "chunk_index": 0, "score": 0.9}
    monkeypatch.setattr(llm_client, "retrieve", lambda q: [hit])
    return store


def _use_fake_client(monkeypatch: pytest.MonkeyPatch, responses: list[Any]) -> FakeCompletions:
    fake = FakeCompletions(responses)
    monkeypatch.setattr(llm_client, "_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=fake)))
    return fake


def test_chat_search_then_book_across_turns(monkeypatch: pytest.MonkeyPatch, fake_env: dict[str, list[Any]]) -> None:
    fake = _use_fake_client(monkeypatch, [
        # turn 1: search, then answer
        _resp(tool_calls=[_tool_call("t1", "search_documents", {"query": "skills"})]),
        _resp(content="Python and SQL (jd.txt)."),
        # turn 2: book (first attempt invalid email, then fixed), then confirm
        _resp(tool_calls=[_tool_call("t2", "book_interview", {
            "name": "Ann Lee", "email": "bad", "date": TOMORROW, "time": "10:00"})]),
        _resp(tool_calls=[_tool_call("t3", "book_interview", {
            "name": "Ann Lee", "email": "ann@example.com", "date": TOMORROW, "time": "10:00"})]),
        _resp(content="Booked!"),
    ])
    with SessionLocal() as db:
        r1 = llm_client.chat("s1", "What skills are needed?", db)
        assert r1.answer == "Python and SQL (jd.txt)." and r1.sources[0]["filename"] == "jd.txt"

        r2 = llm_client.chat("s1", "Book me: Ann Lee, ann@example.com, tomorrow 10am", db)
        assert r2.answer == "Booked!" and r2.booking is not None
        assert db.query(Booking).filter_by(session_id="s1").count() == 1

    # system prompt is prepended; the second-turn request carried turn-1 history
    assert fake.calls[0]["messages"][0]["role"] == "system"
    assert fake.calls[2]["messages"][1]["content"] == "What skills are needed?"
    error_result = fake.calls[3]["messages"][-1]
    assert error_result["role"] == "tool" and error_result["tool_call_id"] == "t2"
    assert error_result["content"].startswith("ERROR:") and "email" in error_result["content"]
    roles = [m["role"] for m in fake_env["s1"]]
    assert roles == [
        "user", "assistant", "tool", "assistant",
        "user", "assistant", "tool", "assistant", "tool", "assistant",
    ]


def test_chat_slot_taken_and_fallback(monkeypatch: pytest.MonkeyPatch, fake_env: dict[str, list[Any]]) -> None:
    args = {"name": "Bob Ray", "email": "bob@example.com", "date": TOMORROW, "time": "15:00"}
    _use_fake_client(monkeypatch, [
        _resp(tool_calls=[_tool_call("a", "book_interview", args)]),
        _resp(tool_calls=[_tool_call("b", "book_interview", args)]),
        _resp(content="That slot is taken."),
        _resp(),  # empty response -> fallback answer
    ])
    with SessionLocal() as db:
        r = llm_client.chat("s2", "book twice", db)
        assert r.answer == "That slot is taken."
        assert llm_client.chat("s2", "bad request", db).answer == llm_client.FALLBACK_ANSWER
    # empty response stored as a plain text exchange so history stays valid
    assert fake_env["s2"][-2:] == [
        {"role": "user", "content": "bad request"},
        {"role": "assistant", "content": llm_client.FALLBACK_ANSWER},
    ]
