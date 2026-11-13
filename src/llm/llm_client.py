"""Conversational RAG with Ollama: a manual tool-use loop over two tools.

- search_documents: custom retrieval against Milvus (no LangChain chains).
- book_interview:   validates and stores a booking.

History lives in Redis; each request appends its new user/assistant/tool messages.
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any

from openai import OpenAI
from pydantic import ValidationError
from sqlalchemy.orm import Session

from src.booking.service import BookingRequest, SlotTakenError, create_booking
from src.config import get_settings
from src.db.models import Booking
from src.memory import chat_memory
from src.prompts.prompt_templates import build_system_prompt
from src.retrieval.retriever import format_context, retrieve
from src.vectordb.vector_store import ChunkHit

log = logging.getLogger(__name__)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Semantic search over the uploaded documents. Returns the most relevant passages.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Standalone search query."}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_interview",
            "description": "Book an interview once name, email, date and time are known and confirmed by the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Candidate's full name."},
                    "email": {"type": "string", "description": "Candidate's email address."},
                    "date": {"type": "string", "description": "Interview date, YYYY-MM-DD."},
                    "time": {"type": "string", "description": "Interview time, 24-hour HH:MM."},
                },
                "required": ["name", "email", "date", "time"],
                "additionalProperties": False,
            },
        },
    },
]

FALLBACK_ANSWER = "Sorry, I couldn't complete that request. Please try rephrasing."


@dataclass
class ChatResult:
    answer: str
    sources: list[ChunkHit] = field(default_factory=list)
    booking: Booking | None = None


@lru_cache
def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(base_url=s.ollama_base_url, api_key=s.ollama_api_key)


def _run_tool(name: str, args: dict[str, Any], session_id: str, db: Session, result: ChatResult) -> tuple[str, bool]:
    """Execute one tool call. Returns (content, is_error)."""
    if name == "search_documents":
        hits = retrieve(args["query"])
        result.sources.extend(hits)
        return format_context(hits), False
    if name == "book_interview":
        try:
            booking = create_booking(db, session_id, BookingRequest.model_validate(args))
        except ValidationError as e:
            return f"Invalid booking details: {e.errors(include_url=False, include_context=False)}", True
        except SlotTakenError as e:
            return str(e), True
        result.booking = booking
        return json.dumps({"status": "booked", "booking_id": booking.id}), False
    return f"Unknown tool: {name}", True


def chat(session_id: str, user_message: str, db: Session) -> ChatResult:
    s = get_settings().llm
    history = chat_memory.load_history(session_id)
    user_turn: dict[str, Any] = {"role": "user", "content": user_message}
    new_messages: list[dict[str, Any]] = [user_turn]
    result = ChatResult(answer=FALLBACK_ANSWER)
    completed = False

    for _ in range(s.max_tool_iterations):
        response = _client().chat.completions.create(
            model=s.model,
            max_tokens=s.max_tokens,
            temperature=s.temperature,
            tools=TOOLS,
            messages=[
                {"role": "system", "content": build_system_prompt(date.today())},
                *history,
                *new_messages,
            ],
        )
        message = response.choices[0].message

        if not message.tool_calls:
            result.answer = (message.content or "").strip() or FALLBACK_ANSWER
            new_messages.append({"role": "assistant", "content": result.answer})
            completed = True
            break

        new_messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                content, is_error = f"Invalid tool arguments: {e}", True
            else:
                content, is_error = _run_tool(call.function.name, args, session_id, db, result)
            log.info("tool=%s error=%s session=%s", call.function.name, is_error, session_id)
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": f"ERROR: {content}" if is_error else content,
                }
            )

    if not completed:
        # Tool-loop limit: store a plain exchange so history stays a valid transcript.
        new_messages = [user_turn, {"role": "assistant", "content": result.answer}]
    chat_memory.append_messages(session_id, new_messages)
    return result
