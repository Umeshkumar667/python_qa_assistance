"""
Python Q&A Assistant — FastAPI Application
Analytics Vidhya AI Engineer Assessment
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    ErrorResponse,
)
from app.rag.pipeline import QAPipeline

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan: load pipeline once at startup ───────────────────────────────────
pipeline: QAPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading RAG pipeline…")
    pipeline = QAPipeline()
    pipeline.load()
    logger.info("RAG pipeline ready.")
    yield
    logger.info("Shutting down.")


# ── OpenAPI tags metadata (used by Swagger UI) ────────────────────────────────
tags_metadata = [
    {
        "name": "system",
        "description": "System-level endpoints (health check, root redirect).",
    },
    {
        "name": "qa",
        "description": "Question-answering endpoint powered by the RAG pipeline.",
    },
]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Python Q&A Assistant By Umeshkumar Pal",
    description=(
        "An AI-powered Q&A system grounded in Stack Overflow Python Q&A data. "
        "Uses a Retrieval-Augmented Generation (RAG) pipeline.\n\n"
        "## Endpoints\n"
        "- **`GET /`** – Redirects to this interactive API documentation.\n"
        "- **`GET /health`** – Health-check endpoint.\n"
        "- **`POST /ask`** – Submit a Python-related question and receive an answer.\n\n"
        "## Try It Out\n"
        "Click the **`POST /ask`** endpoint below, then **Try it out**, "
        "fill in a question, and click **Execute**."
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
    contact={
        "name": "Analytics Vidhya – AI Engineer Assessment",
        "url": "https://www.analyticsvidhya.com",
    },
    license_info={
        "name": "MIT",
    },
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,   # Collapse schemas by default
        "tryItOutEnabled": True,           # Enable Try-It-Out by default
        "displayRequestDuration": True,    # Show request duration
        "syntaxHighlight.theme": "monokai",
        "filter": True,                    # Enable the filter/search box
    },
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request timing ────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.4f}s"
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["system"])
async def root():
    """Redirect to the interactive API documentation."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health-check endpoint."""
    return HealthResponse(
        status="ok",
        model=settings.llm_model,
        vectorstore_loaded=pipeline is not None and pipeline.is_ready(),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["qa"],
)
async def ask(body: AskRequest):
    """
    Answer a Python-related question using the RAG pipeline.

    - **question**: The Python / data-science question to answer.
    - **top_k**: Number of source documents to retrieve (1-10, default 5).
    """
    if pipeline is None or not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not ready. Try again shortly.")

    logger.info("Question: %s", body.question[:120])
    try:
        result = await pipeline.aask(body.question, top_k=body.top_k)
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(**result)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )
