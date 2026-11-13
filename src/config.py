"""Application settings: non-secret values from config.yaml, secrets from .env / environment."""
from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class LLMSettings(BaseModel):
    model: str
    temperature: float
    max_tokens: int
    max_tool_iterations: int


class EmbeddingSettings(BaseModel):
    model: str
    dimension: int


class VectorDBSettings(BaseModel):
    collection: str


class ChunkingSettings(BaseModel):
    chunk_size: int
    chunk_overlap: int


class RetrievalSettings(BaseModel):
    top_k: int


class MemorySettings(BaseModel):
    ttl_seconds: int


class IngestionSettings(BaseModel):
    max_file_mb: int


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", yaml_file="config.yaml", extra="ignore")

    app_name: str
    llm: LLMSettings
    embeddings: EmbeddingSettings
    vectordb: VectorDBSettings
    chunking: ChunkingSettings
    retrieval: RetrievalSettings
    memory: MemorySettings
    ingestion: IngestionSettings

    # Secrets / environment-specific
    milvus_uri: str
    milvus_token: str = ""
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "sqlite:///./app.db"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"  # required by the OpenAI client, ignored by Ollama

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls))


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # populated from yaml/env
