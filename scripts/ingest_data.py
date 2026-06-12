"""
ingest_data.py
==============
Processes the Kaggle Stack Overflow Python Q&A dataset and builds a FAISS
vector store that the API uses at runtime.

Usage
-----
    python scripts/ingest_data.py --questions data/Questions.csv \\
                                   --answers   data/Answers.csv  \\
                                   --limit     50000

The dataset can be downloaded from:
    https://www.kaggle.com/datasets/stackoverflow/pythonquestions

Files needed:
  - Questions.csv
  - Answers.csv

Optional:
  - Tags.csv  (used for metadata, but not required)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Ensure the project root is on sys.path so `app` imports work.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.rag.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── HTML cleaning ─────────────────────────────────────────────────────────────
def to_plain_string(value) -> str:
    """Convert any value to a plain Python string, handling NaN and numpy types."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        if value.lower() == "nan":
            return ""
        return value
    # Handle numpy/pandas scalar types
    import numpy as np
    if isinstance(value, (np.ndarray, np.generic)):
        if hasattr(value, 'item'):
            value = value.item()
        else:
            value = str(value)
    return str(value)


def strip_html(html_text) -> str:
    """Remove HTML tags and normalise whitespace."""
    text = to_plain_string(html_text)
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ").strip()


# ── Load and join ──────────────────────────────────────────────────────────────
def load_data(questions_path: str, answers_path: str, limit: int) -> pd.DataFrame:
    logger.info("Loading questions from %s …", questions_path)
    q_cols = ["Id", "Score", "Title", "Body", "AcceptedAnswerId"]
    questions = pd.read_csv(
        questions_path,
        usecols=[c for c in q_cols if c in pd.read_csv(questions_path, nrows=0, encoding="latin-1").columns],
        nrows=limit,
        on_bad_lines="skip",
        encoding="latin-1",
    )
    # Keep high-quality questions (score ≥ 0)
    if "Score" in questions.columns:
        questions = questions[questions["Score"] >= 0]

    logger.info("Loading answers from %s …", answers_path)
    a_cols = ["Id", "ParentId", "Score", "Body"]
    answers = pd.read_csv(
        answers_path,
        usecols=a_cols,
        on_bad_lines="skip",
        encoding="latin-1",
    )

    # For each question, pick accepted answer first; fall back to highest-score answer.
    logger.info("Joining questions with best answers…")
    best_answers = (
        answers.sort_values("Score", ascending=False)
        .groupby("ParentId")
        .first()
        .reset_index()
        .rename(columns={"ParentId": "QuestionId", "Body": "AnswerBody", "Score": "AnswerScore"})
    )

    merged = questions.merge(
        best_answers[["QuestionId", "AnswerBody", "AnswerScore"]],
        left_on="Id",
        right_on="QuestionId",
        how="left",
    ).dropna(subset=["AnswerBody"])

    logger.info("Retained %d Q&A pairs after join.", len(merged))
    return merged


# ── Build documents ────────────────────────────────────────────────────────────
def build_documents(df: pd.DataFrame) -> list[Document]:
    docs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building docs"):
        title = strip_html(row.get("Title", ""))
        q_body = strip_html(row.get("Body", ""))
        a_body = strip_html(row.get("AnswerBody", ""))

        content = to_plain_string(f"Q: {title}\n\n{q_body}\n\nA: {a_body}")

        if content.strip():
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "question_id": to_plain_string(row.get("Id", "")),
                        "title": title,
                        "question_score": int(row.get("Score", 0)) if not pd.isna(row.get("Score")) else 0,
                        "answer_score": int(row.get("AnswerScore", 0)) if not pd.isna(row.get("AnswerScore")) else 0,
                        "source": "stackoverflow",
                    },
                )
            )
    return docs


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ingest Stack Overflow dataset into FAISS.")
    parser.add_argument("--questions", default="data/Questions.csv", help="Path to Questions.csv")
    parser.add_argument("--answers", default="data/Answers.csv", help="Path to Answers.csv")
    parser.add_argument("--limit", type=int, default=50_000, help="Max questions to load (default 50 000)")
    parser.add_argument("--force", action="store_true", help="Start from scratch even if a partial index exists")
    args = parser.parse_args()

    if not os.path.exists(args.questions):
        logger.error("Questions file not found: %s", args.questions)
        logger.error("Download from https://www.kaggle.com/datasets/stackoverflow/pythonquestions")
        sys.exit(1)
    if not os.path.exists(args.answers):
        logger.error("Answers file not found: %s", args.answers)
        sys.exit(1)

    # Load and join
    df = load_data(args.questions, args.answers, args.limit)

    # Build raw documents
    documents = build_documents(df)
    logger.info("Created %d raw documents.", len(documents))

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if c.page_content and isinstance(c.page_content, str) and c.page_content.strip()]
    logger.info("Split into %d chunks (chunk_size=%d, overlap=%d).",
                len(chunks), settings.chunk_size, settings.chunk_overlap)

    # Build and save vector store
    vs = VectorStore()
    vs.build(chunks, force=args.force)
    logger.info("✅  Ingestion complete. Vector store saved to '%s'.", settings.vectorstore_path)


if __name__ == "__main__":
    main()
