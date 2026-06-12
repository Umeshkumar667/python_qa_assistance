"""
RAG Pipeline
============
1. Retrieve the top-k most similar documents from the FAISS index.
2. Optionally use MMR (Maximal Marginal Relevance) for diversity.
3. Apply similarity threshold to filter low-relevance documents.
4. Optionally rerank results with a cross-encoder model.
5. Build a prompt with the retrieved context.
6. Call the LLM (Groq, OpenAI, or Anthropic) to generate a grounded answer.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful Python teaching assistant for beginners.

Rules:
- Answer ONLY the user's question. Ignore everything else in the context.
- Do NOT explain unrelated concepts from the retrieved context.
- Use retrieved information ONLY when it directly helps answer the question.
- Prioritize concise, educational explanations for beginner Python learners.
- Avoid advanced concepts (metaclasses, decorators, generators, etc.) unless explicitly asked.
- If context contains unrelated information (e.g. metaclasses when asked about classes), DO NOT use it.
- Provide practical code examples only when they help answer the question directly.
- If the context lacks enough relevant information, say so clearly rather than making things up.
- Format your response using Markdown for readability."""

HUMAN_TEMPLATE = """### Context from Stack Overflow
{context}

### Question
{question}

### Answer
"""


def _build_llm():
    """Instantiate the LLM based on the configured provider."""
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
    elif settings.llm_provider == "groq":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )


def _format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "Untitled")
        parts.append(f"[{i}] **{title}**\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _apply_similarity_threshold(
    docs_with_scores: List[Tuple[Document, float]],
    threshold: float,
) -> List[Tuple[Document, float]]:
    """Filter out documents below the similarity threshold."""
    return [(d, s) for d, s in docs_with_scores if s >= threshold]


def _rerank_docs(
    query: str,
    docs_with_scores: List[Tuple[Document, float]],
    top_k: int,
) -> List[Tuple[Document, float, float]]:
    """Rerank documents using a cross-encoder model.

    Returns list of (Document, similarity_score, rerank_score) tuples.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. Skipping reranking. "
            "Install with: pip install sentence-transformers"
        )
        return [(d, s, 0.0) for d, s in docs_with_scores]

    try:
        model = CrossEncoder(settings.rerank_model)
    except Exception as exc:
        logger.warning("Failed to load reranker model '%s': %s. Skipping reranking.", settings.rerank_model, exc)
        return [(d, s, 0.0) for d, s in docs_with_scores]

    pairs = [[query, doc.page_content] for doc, _ in docs_with_scores]
    try:
        scores = model.predict(pairs)
    except Exception as exc:
        logger.warning("Reranking prediction failed: %s. Skipping reranking.", exc)
        return [(d, s, 0.0) for d, s in docs_with_scores]

    # Combine original docs with rerank scores
    results = []
    for (doc, sim_score), rerank_score in zip(docs_with_scores, scores):
        results.append((doc, sim_score, float(rerank_score)))

    # Sort by rerank score descending
    results.sort(key=lambda x: x[2], reverse=True)

    # Keep top_k
    return results[:top_k]


class QAPipeline:
    def __init__(self):
        self._vectorstore = VectorStore()
        self._llm = None
        self._ready = False

    def load(self) -> None:
        """Load the vector store and LLM.  Called once at app startup."""
        ok = self._vectorstore.load()
        if not ok:
            logger.warning(
                "Vector store not ready. The /ask endpoint will return 503 "
                "until ingest_data.py has been run."
            )
        self._llm = _build_llm()
        self._ready = ok

    def is_ready(self) -> bool:
        return self._ready

    # ── Sync ask (for testing) ────────────────────────────────────────────
    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        return asyncio.get_event_loop().run_until_complete(self.aask(question, top_k))

    # ── Async ask (used by FastAPI) ───────────────────────────────────────
    async def aask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        # 1. Retrieve
        t0 = time.perf_counter()

        # Determine effective k for initial retrieval
        fetch_k = max(top_k, settings.mmr_fetch_k) if settings.use_mmr else top_k

        # Choose retrieval method: MMR or plain similarity
        if settings.use_mmr:
            logger.info("Using MMR retrieval (k=%d, fetch_k=%d, lambda_mult=%.2f)", top_k, fetch_k, settings.mmr_lambda_mult)
            docs_with_scores = self._vectorstore.max_marginal_relevance_search_with_score(
                question,
                k=top_k,
                fetch_k=fetch_k,
                lambda_mult=settings.mmr_lambda_mult,
            )
        else:
            docs_with_scores = self._vectorstore.similarity_search_with_score(question, k=top_k)

        retrieval_ms = (time.perf_counter() - t0) * 1000

        # 2. Apply similarity threshold
        docs_with_scores = _apply_similarity_threshold(docs_with_scores, settings.similarity_threshold)
        logger.info(
            "Retrieved %d documents after similarity threshold (>= %.2f)",
            len(docs_with_scores),
            settings.similarity_threshold,
        )

        if not docs_with_scores:
            return {
                "question": question,
                "answer": (
                    "I couldn't find enough relevant information in the knowledge base "
                    "to answer your question. Please try rephrasing or ask a different question."
                ),
                "sources": [],
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": 0.0,
            }

        docs = [d for d, _ in docs_with_scores]
        scores = [s for _, s in docs_with_scores]

        # Abort if the best-matching document is too weak
        max_score = max(scores)
        if max_score < 0.65:
            logger.info("Best score %.4f is below 0.65 — returning 'no relevant info' response.", max_score)
            return {
                "question": question,
                "answer": (
                    "I couldn't find relevant information in the knowledge base "
                    "to answer your question. Please try rephrasing or ask a different question."
                ),
                "sources": [],
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": 0.0,
            }

        # 3. Optionally rerank with cross-encoder
        rerank_scores: List[float] = []
        if settings.use_reranking:
            t_rr = time.perf_counter()
            logger.info("Reranking %d documents with %s", len(docs_with_scores), settings.rerank_model)
            reranked = _rerank_docs(question, docs_with_scores, settings.rerank_top_k)
            rerank_ms = (time.perf_counter() - t_rr) * 1000
            logger.info("Reranking took %.2f ms", rerank_ms)

            docs = [d for d, _, _ in reranked]
            scores = [s for _, s, _ in reranked]
            rerank_scores = [rs for _, _, rs in reranked]
        else:
            rerank_scores = [0.0] * len(docs)

        # 4. Build prompt
        context = _format_context(docs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": HUMAN_TEMPLATE.format(context=context, question=question)},
        ]

        # 5. Generate
        t1 = time.perf_counter()
        from langchain_core.messages import HumanMessage, SystemMessage

        response = await self._llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT),
             HumanMessage(content=HUMAN_TEMPLATE.format(context=context, question=question))]
        )
        generation_ms = (time.perf_counter() - t1) * 1000
        answer = response.content

        # 6. Build source list
        sources = [
            {
                "content": doc.page_content[:500],
                "score": round(float(score), 4),
                "rerank_score": round(float(rerank_score), 4) if rerank_score else None,
                "metadata": doc.metadata,
            }
            for doc, score, rerank_score in zip(docs, scores, rerank_scores)
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
        }