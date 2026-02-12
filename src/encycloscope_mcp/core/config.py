"""Configuration management for Encycloscope MCP."""

import os
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Data paths
    data_dir: Path = Field(default=Path("./data"), description="Directory containing data files")
    articles_path: Path = Field(
        default=Path("./data/articles.csv"),
        description="Path to full articles CSV"
    )

    # New d'AlemBERT + FAISS index (preferred)
    faiss_index_path: Path = Field(
        default=Path("./data/dalembert_index.faiss"),
        description="Path to FAISS index for semantic search"
    )
    chunks_path: Path = Field(
        default=Path("./data/dalembert_index_chunks.pkl"),
        description="Path to chunks metadata for semantic search"
    )

    # Legacy paths (backward compatibility)
    model_path: Path = Field(default=Path("./data/encyclopedia_camembert_aligned"), description="Path to embeddings model (legacy)")
    index_path: Path = Field(default=Path("./data/encyclopedia_index.pkl"), description="Path to search index (legacy)")
    blocks_path: Path = Field(default=Path("./data/encyclopedia_blocks.csv"), description="Path to blocks metadata (legacy)")

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8001, description="Server port")

    # Authentication
    mcp_token: str | None = Field(default=None, description="Optional authentication token")

    # CORS
    cors_allowed_origins: str = Field(default="*", description="Allowed CORS origins")

    # Search parameters
    top_k: int = Field(default=10, description="Default number of results")
    similarity_threshold: float = Field(default=0.5, description="Minimum similarity score")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
