from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"
    qdrant_path: Path = Path("backend/data/qdrant")
    qdrant_collection: str = "transcript_chunks"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
