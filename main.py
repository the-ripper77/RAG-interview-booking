import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from src.api.routes import router
from src.config import get_settings
from src.db.models import init_db

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/app.log", encoding="utf-8")],
)
logging.getLogger("watchfiles").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title=get_settings().app_name, lifespan=lifespan)
app.include_router(router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_excludes=["logs/*", "data/*", "*.db"])
