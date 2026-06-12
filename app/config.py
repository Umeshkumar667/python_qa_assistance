"""
Centralised configuration — all values come from environment variables
(or a .env file when running locally).
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM ───────────────────────────────────────────────────────────────────
    # Set LLM_PROVIDER to "groq" (default), "openai", or "anthropic"
    llm_provider: str = "groq"

    # Groq (default)
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # OpenAI
    openai_api_key: str = ""

    # Anthropic
    anthropic_api_key: str = ""

    # Model names
    llm_model: str = "openai/gpt-oss-20b"          # Groq default
    anthropic_model: str = "claude-haiku-4-5-20251001"  # Anthropic alternative

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: str = "huggingface"         # "huggingface" (default) | "openai"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # ── Vector store ─────────────────────────────────────────────────────────
    vectorstore_path: str = "data/vectorstore"   # FAISS index directory
    vectorstore_top_k: int = 5

    # ── RAG ───────────────────────────────────────────────────────────────────
    max_tokens: int = 1024
    temperature: float = 0.2
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # ── Retrieval ─────────────────────────────────────────────────────────────
    similarity_threshold: float = 0.75  # Minimum similarity score for retrieved docs

    # ── MMR ───────────────────────────────────────────────────────────────────
    use_mmr: bool = False               # Set True to enable MMR retrieval
    mmr_fetch_k: int = 20               # Number of docs to fetch before MMR selection
    mmr_lambda_mult: float = 0.5        # MMR diversity (0 = max diversity, 1 = max relevance)

    # ── Reranking ─────────────────────────────────────────────────────────────
    use_reranking: bool = False         # Set True to enable cross-encoder reranking
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder model
    rerank_top_k: int = 3               # Number of docs to keep after reranking


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()