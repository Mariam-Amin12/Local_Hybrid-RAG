from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # OpenAI API key
    OPENAI_API_KEY: str = "YOUR_OPENAI_API_KEY"
    # RAG pipeline parameters
    RAG_PERSIST_DIR: str = "./rag_store"
    RAG_CHUNK_SIZE: int = 500
    RAG_OVERLAP: int = 88
    RAG_TOP_K: int = 20
    RAG_TOP_N: int = 5
    ENV : str = "development"  # or "production"

    API_KEY: str = "dev-key-123"
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

@lru_cache()
def get_settings() -> Settings: # 3alashan a read it only once 
    settings = Settings()
    print(
        f"[config] Loaded environment={settings.ENV}, "
        f"persist_dir={settings.RAG_PERSIST_DIR}",
        flush=True,
    )
    return settings