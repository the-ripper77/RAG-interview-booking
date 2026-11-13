"""Redis-backed chat memory: one list per session, each item a JSON-encoded API message."""
import json
from functools import lru_cache
from typing import Any

import redis

from src.config import get_settings

Message = dict[str, Any]


@lru_cache
def _redis() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def _key(session_id: str) -> str:
    return f"chat:v2:{session_id}"


def load_history(session_id: str) -> list[Message]:
    return [json.loads(m) for m in _redis().lrange(_key(session_id), 0, -1)]


def append_messages(session_id: str, messages: list[Message]) -> None:
    # ponytail: history is never trimmed; sessions expire via TTL. Add server-side compaction
    # if conversations get very long.
    if not messages:
        return
    key = _key(session_id)
    pipe = _redis().pipeline()
    pipe.rpush(key, *(json.dumps(m) for m in messages))
    pipe.expire(key, get_settings().memory.ttl_seconds)
    pipe.execute()


def clear(session_id: str) -> None:
    _redis().delete(_key(session_id))
