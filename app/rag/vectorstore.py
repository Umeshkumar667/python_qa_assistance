"""
FAISS-based vector store.

Responsibilities:
- Build the index from LangChain Documents
- Persist to / load from disk
- Similarity search with scores
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)

# Metadata file that records how many batches have been persisted so far.
_RESUME_META = "resume_meta.json"


def _resume_meta_path() -> str:
    return str(Path(settings.vectorstore_path) / _RESUME_META)


def _load_resume_meta() -> dict | None:
    path = _resume_meta_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _save_resume_meta(meta: dict) -> None:
    path = _resume_meta_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)


def _clear_resume_meta() -> None:
    try:
        os.remove(_resume_meta_path())
    except FileNotFoundError:
        pass


def _build_embeddings():
    """Build embeddings based on the configured embedding provider."""
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    else:
        # Default: HuggingFace sentence-transformers (runs locally, no API key needed)
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


class VectorStore:
    """Thin wrapper around a FAISS index."""

    def __init__(self):
        self._embeddings = _build_embeddings()
        self._store: FAISS | None = None

    # ── Build / Persist ────────────────────────────────────────────────────
    def build(
        self,
        documents: List[Document],
        batch_size: int = 500,
        force: bool = False,
    ) -> None:
        """Create a FAISS index from `documents`, saving after each batch.

        Behaviour
        ---------
        * If a complete index already exists on disk **and** ``force`` is
          ``False``, the existing index is loaded immediately and no documents
          are re-processed.
        * If a **partial** index (from a previous crash or Ctrl-C) exists,
          embedding resumes from the next unprocessed batch.
        * After every successfully embedded batch the FAISS index **and** a
          small ``resume_meta.json`` checkpoint are persisted to
          ``settings.vectorstore_path``.
        * On successful completion the ``resume_meta.json`` is updated with a
          ``"completed": true`` marker so subsequent runs can skip straight to
          loading the saved index.
        """
        # ── Fast path: skip if a complete index already exists ─────────────────────
        if not force:
            path = Path(settings.vectorstore_path)
            if path.is_dir() and any(path.iterdir()):
                try:
                    store = FAISS.load_local(
                        str(path),
                        self._embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    logger.info(
                        "✅ Index already exists at %s (%d vectors). "
                        "Use --force to rebuild from scratch.",
                        settings.vectorstore_path,
                        store.index.ntotal,
                    )
                    self._store = store
                    return
                except Exception:
                    logger.info(
                        "Existing index at %s is corrupt or incompatible – will rebuild.",
                        settings.vectorstore_path,
                    )

        # Validate and clean documents ---------------------------------------------------
        raw_texts: list[str] = []
        raw_metas: list[dict] = []

        for doc in documents:
            content = doc.page_content
            if content is None:
                continue

            # Convert to native Python string - handles numpy/pandas string types
            try:
                content = str(str(content).strip()) if content is not None else None
            except Exception:
                continue

            if not content or content.lower() == "nan":
                continue

            clean_meta: dict = {}
            for key, value in doc.metadata.items():
                if value is None or (
                    isinstance(value, float) and str(value).lower() == "nan"
                ):
                    clean_meta[key] = ""
                else:
                    clean_meta[key] = str(value)

            raw_texts.append(content)
            raw_metas.append(clean_meta)

        if not raw_texts:
            raise ValueError("No valid documents to embed after filtering")

        total = len(raw_texts)
        logger.info("Building FAISS index from %d documents (batch_size=%d)…", total, batch_size)

        # --- Resume logic ----------------------------------------------------------------
        start_batch = 0
        if not force:
            existing = self._try_load_existing(total)
            if existing is not None:
                self._store, start_batch = existing
                logger.info(
                    "Resumed from existing index at batch %d/%d.",
                    start_batch,
                    (total + batch_size - 1) // batch_size,
                )

        # --- Embed & persist batch by batch ---------------------------------------------
        for batch_idx, start in enumerate(range(start_batch * batch_size, total, batch_size)):
            end = min(start + batch_size, total)

            batch_texts: list[str] = []
            batch_metas: list[dict] = []
            for i in range(start, end):
                text = raw_texts[i]
                # Ensure native Python string - handle numpy/pandas string types
                if text is None:
                    logger.warning("Skipping document at index %d: None content.", i)
                    continue
                # Use repr to handle non-native string types, then check validity
                try:
                    text = str(str(text).strip())
                except Exception:
                    logger.warning("Skipping document at index %d: cannot convert to string.", i)
                    continue
                if not text or text.lower() == "nan":
                    logger.warning("Skipping document at index %d: empty or NaN-like content.", i)
                    continue
                batch_texts.append(text)
                batch_metas.append(raw_metas[i])

            if not batch_texts:
                logger.warning("Skipping batch %d-%d: no valid documents after filtering.", start, end)
                continue

            current_batch_num = batch_idx + start_batch

            logger.info(
                "Embedding batch %d/%d (docs %d-%d)…",
                current_batch_num + 1,
                (total + batch_size - 1) // batch_size,
                start + 1,
                end,
            )

            try:
                self._embeddings.embed_documents(batch_texts)
            except Exception as exc:
                logger.exception("Failed to embed batch %d-%d: %s", start, end, exc)
                continue

            # Start a fresh index if we don't have one yet, otherwise add to it
            if self._store is None:
                self._store = FAISS.from_texts(batch_texts, self._embeddings, metadatas=batch_metas)
            else:
                self._store.add_texts(batch_texts, metadatas=batch_metas)

            self._save()
            _save_resume_meta(
                {
                    "total_documents": total,
                    "batch_size": batch_size,
                    "processed_batches": current_batch_num + 1,
                    "embedding_model": settings.embedding_model,
                    "embedding_provider": settings.embedding_provider,
                }
            )
            logger.info("  ✓ Saved index after batch %d.", current_batch_num + 1)

        if self._store is None:
            raise RuntimeError("Index was not built – no documents were processed.")

        # Mark as complete
        _clear_resume_meta()
        self._save()
        logger.info(
            "✅  Index built (%d vectors) and saved to %s",
            self._store.index.ntotal,
            settings.vectorstore_path,
        )

    def _try_load_existing(self, total_documents: int) -> tuple[FAISS, int] | None:
        """Try to load a partial / complete index for resume.

        Returns ``(store, processed_batches)`` when a valid checkpoint exists,
        otherwise ``None``.
        """
        path = Path(settings.vectorstore_path)
        meta = _load_resume_meta()
        if meta is None:
            return None

        # Validate checkpoint is still compatible with current config
        if meta.get("embedding_model") != settings.embedding_model:
            logger.warning(
                "Checkpoint uses embedding model %r but current is %r – starting fresh.",
                meta.get("embedding_model"),
                settings.embedding_model,
            )
            return None
        if meta.get("embedding_provider") != settings.embedding_provider:
            logger.warning("Embedding provider changed – starting fresh.")
            return None
        if meta.get("total_documents") != total_documents:
            logger.warning(
                "Checkpoint was built for %d documents but current run has %d – starting fresh.",
                meta.get("total_documents"),
                total_documents,
            )
            return None

        if not os.path.isdir(path) or not os.listdir(path):
            logger.warning("Checkpoint metadata found but no index files on disk – starting fresh.")
            return None

        try:
            store = FAISS.load_local(
                str(path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            logger.exception("Failed to load partial index – starting fresh.")
            return None

        processed = int(meta.get("processed_batches", 0))
        logger.info(
            "Loaded partial index: %d vectors from %d batches.",
            store.index.ntotal,
            processed,
        )
        return store, processed

    def _save(self) -> None:
        Path(settings.vectorstore_path).mkdir(parents=True, exist_ok=True)
        self._store.save_local(settings.vectorstore_path)

    # ── Load ───────────────────────────────────────────────────────────────
    def load(self) -> bool:
        """Load an existing index from disk.  Returns True on success."""
        path = settings.vectorstore_path
        if not os.path.isdir(path) or not os.listdir(path):
            logger.warning("No vectorstore found at %s. Run ingest_data.py first.", path)
            return False
        try:
            self._store = FAISS.load_local(
                path,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded FAISS index from %s (%d vectors)", path, self._store.index.ntotal)
            return True
        except Exception:
            logger.exception("Failed to load FAISS index")
            return False

    # ── Search ─────────────────────────────────────────────────────────────
    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        if self._store is None:
            raise RuntimeError("Vector store not loaded. Call load() or build() first.")
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def max_marginal_relevance_search_with_score(
        self, query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """Perform MMR search returning docs with relevance scores.

        MMR balances relevance (high similarity to query) and diversity
        (minimising similarity among selected documents).

        Returns (Document, similarity_score) tuples where similarity_score
        is the cosine similarity between the query and each returned doc.
        """
        if self._store is None:
            raise RuntimeError("Vector store not loaded. Call load() or build() first.")

        # First, get all candidates with their scores via similarity search
        candidates = self._store.similarity_search_with_relevance_scores(
            query, k=fetch_k
        )

        # Use MMR to select diverse subset
        mmr_docs = self._store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

        # Build a lookup from MMR-selected docs to their similarity scores
        # Match on page_content since Document objects may differ by metadata
        mmr_content_set = {doc.page_content for doc in mmr_docs}
        scored: List[Tuple[Document, float]] = []
        for doc, score in candidates:
            if doc.page_content in mmr_content_set:
                scored.append((doc, score))
                # Remove to avoid duplicates (shouldn't happen, but be safe)
                mmr_content_set.remove(doc.page_content)

        return scored

    def is_ready(self) -> bool:
        return self._store is not None
