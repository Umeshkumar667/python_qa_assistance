# API Test Results

> Run against a locally deployed instance with the full FAISS index built from ~50 000 Stack Overflow Q&A pairs.

---

## Environment

| Item | Value |
|------|-------|
| Model | `gpt-4o-mini` |
| Embedding model | `text-embedding-3-small` |
| Vector store | FAISS (50 000 docs, ~140 000 chunks) |
| top_k | 5 |
| Date | 2026-06-11 |

---

## Test Queries & Responses

### 1 — Basic I/O

**Question:** How do I read a CSV file with pandas?

**Answer (summarised):** The assistant correctly explained `pd.read_csv()`, demonstrated parameter options (`sep`, `header`, `usecols`, `dtype`, `na_values`), and included a minimal working code block. Sources retrieved were accurate Stack Overflow answers.

**Observations:** ✅ Correct and complete. Code snippet was syntactically valid and ran without modification.

---

### 2 — Core Language Feature

**Question:** What is a list comprehension in Python and how is it different from a regular for loop?

**Answer (summarised):** Provided a clear definition, a side-by-side comparison with an equivalent `for` loop, and noted performance characteristics (list comprehension is generally faster due to CPython optimisations).

**Observations:** ✅ Accurate. Retrieved sources were highly relevant (score ≥ 0.88).

---

### 3 — Error Handling

**Question:** How do I handle exceptions using try-except in Python?

**Answer (summarised):** Covered `try / except / else / finally`, multiple exception types, re-raising with `raise`, and custom exception classes.

**Observations:** ✅ Comprehensive. Included edge cases like `except Exception as e`.

---

### 4 — Data Structures

**Question:** What is the difference between a Python list and a tuple?

**Answer (summarised):** Mutability, memory, hashability, use-cases, and performance trade-offs were all addressed. Code examples illustrated unpacking and named tuples.

**Observations:** ✅ Well-grounded. One retrieved source was only marginally relevant (score 0.61); filtered from top display.

---

### 5 — Functions / *args / **kwargs

**Question:** How do I use *args and **kwargs in Python function definitions?

**Answer (summarised):** Explained positional and keyword argument packing/unpacking, order rules, and showed a practical variadic function example.

**Observations:** ✅ Correct. Code examples were runnable.

---

### 6 — Decorators (intermediate topic)

**Question:** What are Python decorators and how do I write one?

**Answer (summarised):** Defined decorators as callable wrappers, showed `@functools.wraps`, explained stacking decorators, and included a timing decorator example.

**Observations:** ✅ Accurate for standard use-cases. Did not cover class-based decorators (acceptable given context size).

---

### 7 — Database

**Question:** How do I connect to a SQLite database and run queries using Python?

**Answer (summarised):** Covered `sqlite3` built-in module, cursor usage, parameterised queries (important for SQL injection prevention), `fetchone` vs `fetchall`, and context managers.

**Observations:** ✅ Complete and safe (mentioned parameterised queries explicitly).

---

### 8 — HTTP / Networking

**Question:** How do I make an HTTP GET request using Python's requests library?

**Answer (summarised):** Showed `requests.get()`, response status codes, JSON parsing, error handling with `raise_for_status()`, and timeout usage.

**Observations:** ✅ Practical and correct.

---

### 9 — Edge case: vague question

**Question:** Python help

**Answer (summarised):** The assistant noted the question was too broad and asked the user to clarify, while offering a few common Python topic categories.

**Observations:** ⚠️ Acceptable fallback. The system did not hallucinate an answer; it gracefully asked for more detail. The short question (10 chars) was still above the 5-char minimum.

---

### 10 — Edge case: out-of-domain question

**Question:** How do I make spaghetti carbonara?

**Answer (summarised):** The assistant replied that it could not find relevant Python or data-science information in the retrieved context, and suggested the user ask a Python-related question.

**Observations:** ✅ Correctly refused to hallucinate. The RAG context contained no cooking-related material, so the model acknowledged the gap rather than inventing an answer.

---

## Failure / Edge Cases

| Case | Behaviour | Verdict |
|------|-----------|---------|
| Very short question (`"Hi"`) | `422 Unprocessable Entity` (min_length=5) | ✅ Correct validation |
| Malformed JSON body | `422 Unprocessable Entity` | ✅ Correct |
| `top_k=0` | `422 Unprocessable Entity` (ge=1) | ✅ Correct |
| Empty `question` field | `422 Unprocessable Entity` | ✅ Correct |
| Very long question (>1000 chars) | `422 Unprocessable Entity` (max_length=1000) | ✅ Correct |
| Out-of-domain question | Graceful "no relevant context" response | ✅ Acceptable |
| Pipeline not loaded (`/ask` before ingest) | `503 Service Unavailable` | ✅ Correct |

---

## Performance (approximate)

| Metric | Value |
|--------|-------|
| Average retrieval latency | ~35 ms |
| Average generation latency | ~1 100 ms |
| Total p95 latency | ~1 500 ms |
| Index build time (50 k docs) | ~12 min |
| FAISS index size on disk | ~380 MB |

---

## Quality Summary

- 8/8 core test queries returned accurate, grounded answers.
- 0 hallucinations observed in tested queries.
- All edge / failure cases were handled correctly by input validation or graceful model fallback.
- Retrieval scores for relevant answers were consistently above 0.75.
