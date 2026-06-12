"""Pydantic schemas for API request and response bodies."""

from typing import List, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        examples=["How do I read a CSV file with pandas?"],
        description="A Python or data-science related question.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of source chunks to retrieve.",
    )

    model_config = {"json_schema_extra": {"example": {"question": "How do I read a CSV file with pandas?", "top_k": 5}}}


class SourceDocument(BaseModel):
    content: str = Field(description="Relevant excerpt from the source document.")
    score: float = Field(description="Cosine-similarity score (higher = more relevant).")
    rerank_score: Optional[float] = Field(default=None, description="Cross-encoder reranking score (higher = more relevant). Only present when reranking is enabled.")
    metadata: dict = Field(default_factory=dict, description="Source metadata (question ID, title, etc.).")


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument] = Field(default_factory=list)
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    vectorstore_loaded: bool


class ErrorResponse(BaseModel):
    error: str
    status_code: int
