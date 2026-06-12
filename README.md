# 🐍 Python Q&A Assistant

An AI-powered question-answering system grounded in Stack Overflow Python Q&A data.
Built with **FastAPI**, **LangChain**, **FAISS**, and **Groq** (or **OpenAI** / **Anthropic Claude**).

---

## 🏗️ Architecture

```
User question
     │
     ▼
 FastAPI  (/ask)
     │
     ▼
 RAG Pipeline
  ├─ 1. Embed question    (BAAI/bge-small-en-v1.5 via HuggingFace)
  ├─ 2. Retrieve top-k    (FAISS similarity search — or MMR)
  ├─ 3. Filter            (similarity threshold + max-score guard)
  ├─ 4. Rerank (opt.)     (cross-encoder for relevance re-scoring)
  └─ 5. Generate answer   (openai/gpt-oss-20b via Groq)
     │
     ▼
 Grounded answer + sources
```

**Data flow (ingestion)**

```
Questions.csv + Answers.csv
        │
        ▼ scripts/ingest_data.py
  Join + clean HTML
        │
        ▼
  Chunk (RecursiveCharacterTextSplitter)
        │
        ▼
   Embed (BAAI/bge-small-en-v1.5)
        │
        ▼
  FAISS index  →  data/vectorstore/
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/python-qa-assistant.git
cd python-qa-assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY (required for Groq provider)
```

### 3. Download the dataset

Download from Kaggle: https://www.kaggle.com/datasets/stackoverflow/pythonquestions

Place `Questions.csv` and `Answers.csv` inside the `data/` directory.

### 4. Build the vector store

```bash
python scripts/ingest_data.py --questions data/Questions.csv \
                               --answers   data/Answers.csv  \
                               --limit     50000
```

> ⏱ This takes ~12 minutes. Local HuggingFace embeddings (BAAI/bge-small-en-v1.5) are used — no API key required for embedding.

### 5. Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for interactive Swagger API documentation.

> 💡 **The root URL (`/`) also redirects to `/docs`** — just open `http://localhost:8000` in your browser.

---

## 🔧 Configuration

All configuration is managed through environment variables (set in `.env`).

### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | One of `groq`, `openai`, `anthropic` |
| `GROQ_API_KEY` | — | Groq API key (required for Groq) |
| `OPENAI_API_KEY` | — | OpenAI API key (required for OpenAI) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (required for Anthropic) |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model name for Groq/OpenAI |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Model name for Anthropic |

### Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` (local, free) or `openai` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model name |

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | `0.75` | Minimum similarity score for a document to be included |
| `VECTORSTORE_TOP_K` | `5` | Number of documents to retrieve |

### MMR (Maximal Marginal Relevance)

Set `USE_MMR=true` to enable MMR-based retrieval, which improves diversity.

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MMR` | `False` | Enable MMR retrieval |
| `MMR_FETCH_K` | `20` | Number of candidates to fetch before MMR selection |
| `MMR_LAMBDA_MULT` | `0.5` | Diversity control: `0` = max diversity, `1` = max relevance |

### Reranking

Set `USE_RERANKING=true` to enable cross-encoder reranking, which re-scores retrieved documents for higher precision.

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_RERANKING` | `False` | Enable cross-encoder reranking |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model name |
| `RERANK_TOP_K` | `3` | Number of documents to keep after reranking |

### RAG

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_TOKENS` | `1024` | Maximum tokens in LLM response |
| `TEMPERATURE` | `0.2` | LLM temperature (lower = more deterministic) |

### Example `.env` with all features enabled

```
# — LLM
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
LLM_MODEL=openai/gpt-oss-20b

# — Retrieval
SIMILARITY_THRESHOLD=0.75

# — MMR
USE_MMR=true
MMR_FETCH_K=20
MMR_LAMBDA_MULT=0.5

# — Reranking
USE_RERANKING=true
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_K=3
```

---

## 🔍 Retrieval Pipeline (step by step)

```
Question → Embed → FAISS search (top-k)
     │
     ▼
 ┌─ Similarity threshold ─────────────────┐
 │  Drop docs with score < 0.75           │
 └────────────────────────────────────────┘
     │
     ▼
 ┌─ Max-score guard ──────────────────────┐
 │  If best doc score < 0.65 → abort      │
 └────────────────────────────────────────┘
     │
     ▼ (optional)
 ┌─ MMR ──────────────────────────────────┐
 │  Re-rank for diversity instead of       │
 │  pure similarity                        │
 └────────────────────────────────────────┘
     │
     ▼ (optional)
 ┌─ Cross-encoder reranking ──────────────┐
 │  Re-score docs, keep top 3             │
 └────────────────────────────────────────┘
     │
     ▼
 Build prompt → LLM → Answer
```

---

## 📡 API Reference

### `GET /health`

Returns the health status of the API and whether the vector store is loaded.

**Response**
```json
{
  "status": "ok",
  "model": "openai/gpt-oss-20b",
  "vectorstore_loaded": true
}
```

---

### `POST /ask`

Answer a Python-related question using the RAG pipeline.

**Request body**
```json
{
  "question": "How do I read a CSV file with pandas?",
  "top_k": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | — | The question (5–1 000 chars) |
| `top_k` | int | 5 | Number of source chunks to retrieve (1–10) |

**Response**
```json
{
  "question": "How do I read a CSV file with pandas?",
  "answer": "You can use `pd.read_csv()`...",
  "sources": [
    {
      "content": "Q: Read CSV with pandas...",
      "score": 0.9123,
      "rerank_score": 2.456,
      "metadata": { "title": "...", "question_id": "123" }
    }
  ],
  "retrieval_ms": 32.5,
  "generation_ms": 1140.0
}
```

> ℹ️ `rerank_score` is only present when reranking is enabled. It represents the cross-encoder's relevance score (higher = more relevant).

---

## 💬 Sample Questions & Answers

### Basic Python

| Question | Answer highlights |
|----------|-------------------|
| `"What is a list comprehension in Python?"` | Concise explanation with syntax example and comparison to for-loops |
| `"How do I reverse a string?"` | `[::-1]` slicing, `reversed()`, and a practical example |
| `"What are *args and **kwargs?"` | Beginner-friendly explanation with function examples |
| `"How do I handle exceptions?"` | `try`/`except`/`finally` with a division-by-zero example |

### Pandas & Data Science

| Question | Answer highlights |
|----------|-------------------|
| `"How do I read a CSV file with pandas?"` | `pd.read_csv()` with path, header, and encoding options |
| `"How do I handle missing values in a DataFrame?"` | `dropna()`, `fillna()`, and interpolation |
| `"How to group data in pandas?"` | `groupby()` with aggregation examples |
| `"How do I merge two DataFrames?"` | `merge()` with different join types |

### Python OOP

| Question | Answer highlights |
|----------|-------------------|
| `"What are classes in Python?"` | Blueprint analogy, `__init__`, `self`, creating objects |
| `"What is inheritance?"` | Parent/child classes, `super()`, method overriding |

> All answers are grounded in Stack Overflow context and tailored for beginners.

---

## 🧪 Running Tests

```bash
# Unit tests (no server required, no API calls)
pytest tests/ -m "not integration" -v

# Integration tests (requires running server)
API_BASE_URL=http://localhost:8000 pytest tests/ -m integration -v

# With coverage
pytest tests/ -m "not integration" --cov=app --cov-report=term-missing
```

---

## 🐳 Docker

```bash
# 1. Copy and fill in your .env
cp .env.example .env

# 2. Build & start
docker-compose up --build

# 3. (First time) run ingestion inside the container
docker-compose exec api python scripts/ingest_data.py \
    --questions data/Questions.csv \
    --answers   data/Answers.csv
```

---

## 👥 Concurrent Users

With a **single Uvicorn worker** (the default), the API handles approximately **2–5 concurrent users**. This is sufficient for personal use or a small team.

### Why the limit?

| Component | Type | Blocks event loop? |
|-----------|------|--------------------|
| **FAISS search** | CPU-bound (vector math) | ✅ Yes — blocks other requests |
| **Embedding** | CPU-bound (transformer inference) | ✅ Yes — blocks other requests |
| **Cross-encoder reranking** | CPU-bound (transformer inference) | ✅ Yes — blocks other requests |
| **LLM call** | I/O-bound (HTTP to Groq/OpenAI) | ❌ No — async, non-blocking |

The CPU-bound operations (FAISS, embeddings, reranking) run synchronously. Only one request can perform these operations at a time per worker.

### Scaling up

| Setup | Concurrent users | Notes |
|-------|-----------------|-------|
| **1 worker** (default) | **2–5** | Fine for personal use or a small team |
| **4 workers** (`--workers 4`) | **10–20** | Each worker runs independently; 4 searches in parallel |
| **8+ workers** | **20–40** | Requires multi-core machine; each worker loads FAISS (~2–3 GB RAM) |

```bash
# Run with 4 workers for 10–20 concurrent users
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

> ⚠️ Each worker loads its own FAISS index + model into memory (~2–3 GB per worker). For higher concurrency, consider a vector database (Pinecone, Qdrant) so all workers share one search service.

---

## 📁 Project Structure

```
python-qa-assistant/
├── app/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Environment-based settings
│   ├── models/
│   │   └── schemas.py    # Pydantic request / response models
│   └── rag/
│       ├── pipeline.py   # Retrieval + generation logic
│       └── vectorstore.py# FAISS wrapper
├── scripts/
│   └── ingest_data.py    # One-time data ingestion
├── tests/
│   ├── test_api.py       # pytest test suite
│   └── test_results.md   # Documented test results
├── data/
│   └── README.md         # Dataset download instructions
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🌐 Deployment Options

### Render (Free tier)

1. Push your repo to GitHub.
2. Go to https://render.com → New Web Service → connect your repo.
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables from your `.env` under "Environment".
5. Use a **Persistent Disk** mounted at `/app/data` and run ingestion once.

### Railway

1. `railway login && railway init`
2. `railway up`
3. Set environment variables in the Railway dashboard.

### Hugging Face Spaces (Docker)

1. Create a new Space with **Docker** as the SDK.
2. Push your repo — Hugging Face will build from `Dockerfile`.
3. Add secrets (GROQ_API_KEY etc.) in the Space settings.

---

## 📄 License

MIT