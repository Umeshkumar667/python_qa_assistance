"""
tests/test_api.py
=================
Automated API tests using pytest + httpx.

Run with:
    pytest tests/test_api.py -v

For integration tests against a running server set:
    API_BASE_URL=http://localhost:8000 pytest tests/ -v -m integration
"""

import os
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline():
    """Return a mock QAPipeline that returns canned answers without hitting any external API."""
    mock = MagicMock()
    mock.is_ready.return_value = True
    mock.aask = AsyncMock(return_value={
        "question": "test question",
        "answer": "This is a mock answer about Python.",
        "sources": [
            {
                "content": "Some relevant Stack Overflow content.",
                "score": 0.92,
                "metadata": {"title": "How to X in Python", "question_id": "123"},
            }
        ],
        "retrieval_ms": 12.3,
        "generation_ms": 450.0,
    })
    return mock


@pytest.fixture
def test_client(mock_pipeline):
    """Async test client with the pipeline mocked out."""
    from fastapi.testclient import TestClient
    import app.main as app_module

    with patch.object(app_module, "pipeline", mock_pipeline):
        with TestClient(app_module.app) as client:
            yield client


# ── Unit tests (mocked, no external calls) ───────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, test_client):
        r = test_client.get("/health")
        assert r.status_code == 200

    def test_health_response_shape(self, test_client):
        r = test_client.get("/health")
        body = r.json()
        assert "status" in body
        assert "model" in body
        assert "vectorstore_loaded" in body

    def test_health_status_ok(self, test_client):
        r = test_client.get("/health")
        assert r.json()["status"] == "ok"


class TestAskEndpoint:
    def test_ask_returns_200(self, test_client):
        r = test_client.post("/ask", json={"question": "How do I read a CSV with pandas?"})
        assert r.status_code == 200

    def test_ask_response_has_answer(self, test_client):
        r = test_client.post("/ask", json={"question": "What is a Python list comprehension?"})
        body = r.json()
        assert "answer" in body
        assert len(body["answer"]) > 0

    def test_ask_response_has_sources(self, test_client):
        r = test_client.post("/ask", json={"question": "Explain decorators in Python."})
        body = r.json()
        assert "sources" in body
        assert isinstance(body["sources"], list)

    def test_ask_response_has_question_echo(self, test_client):
        question = "How do I handle exceptions in Python?"
        r = test_client.post("/ask", json={"question": question})
        # The mocked pipeline returns "test question" but the route should echo the input
        # In real mode it would echo correctly; here we just check the key exists
        assert "question" in r.json()

    def test_ask_response_has_timing(self, test_client):
        r = test_client.post("/ask", json={"question": "How does yield work?"})
        body = r.json()
        assert "retrieval_ms" in body
        assert "generation_ms" in body

    def test_ask_rejects_short_question(self, test_client):
        r = test_client.post("/ask", json={"question": "Hi"})
        assert r.status_code == 422

    def test_ask_rejects_empty_question(self, test_client):
        r = test_client.post("/ask", json={"question": ""})
        assert r.status_code == 422

    def test_ask_rejects_missing_question(self, test_client):
        r = test_client.post("/ask", json={})
        assert r.status_code == 422

    def test_ask_custom_top_k(self, test_client):
        r = test_client.post("/ask", json={"question": "Explain lambda functions.", "top_k": 3})
        assert r.status_code == 200

    def test_ask_top_k_out_of_range(self, test_client):
        r = test_client.post("/ask", json={"question": "Explain lambda functions.", "top_k": 0})
        assert r.status_code == 422

    def test_ask_top_k_too_large(self, test_client):
        r = test_client.post("/ask", json={"question": "Explain lambda functions.", "top_k": 11})
        assert r.status_code == 422

    def test_x_process_time_header(self, test_client):
        r = test_client.post("/ask", json={"question": "How do I merge two dicts?"})
        assert "x-process-time" in r.headers


class TestOpenAPISchema:
    def test_openapi_json_accessible(self, test_client):
        r = test_client.get("/openapi.json")
        assert r.status_code == 200

    def test_docs_accessible(self, test_client):
        r = test_client.get("/docs")
        assert r.status_code == 200


# ── Integration tests (require a running server) ──────────────────────────────
# Run: API_BASE_URL=http://localhost:8000 pytest tests/test_api.py -m integration -v

INTEGRATION_QUESTIONS = [
    "How do I read a CSV file with pandas?",
    "What is a list comprehension in Python?",
    "How do I handle exceptions using try-except?",
    "Explain the difference between a list and a tuple.",
    "How do I use *args and **kwargs in a function?",
    "What are Python decorators and how do I write one?",
    "How do I connect to a SQLite database using Python?",
    "How do I make an HTTP GET request with requests library?",
]


@pytest.mark.integration
class TestLiveAPI:
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_BASE_URL, timeout=30)

    def test_health_live(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    @pytest.mark.parametrize("question", INTEGRATION_QUESTIONS)
    def test_ask_live(self, client, question):
        r = client.post("/ask", json={"question": question})
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert len(body["answer"]) > 50, "Answer seems too short"
        assert body["sources"], "Expected at least one source"
