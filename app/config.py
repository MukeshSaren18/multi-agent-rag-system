from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "enterprise_knowledge"
    chroma_persist_dir: str = "./chroma_data"

    # Retrieval
    top_k_results: int = 5
    similarity_threshold: float = 0.75

    # Server
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"

    # Cache
    enable_prompt_cache: bool = True
    cache_ttl_seconds: int = 300

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
