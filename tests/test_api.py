# tests/test_api.py
import json

from fastapi.testclient import TestClient

from src.api.main import app, get_prompt_runner
from src.baseline.llm_client import MockLLMClient
from src.baseline.prompt_runner import PromptRunner


# override dependency to ensure deterministic MockLLM used in tests
def _get_mock_runner():
    return PromptRunner(llm_client=MockLLMClient(salt="test-salt"))


app.dependency_overrides[get_prompt_runner] = _get_mock_runner

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert "git_sha" in data


def test_grade_endpoint_basic():
    payload = {
        "essay_id": "e100",
        "full_text": "This is a test essay that reasonably argues a point.",
        "assignment": "asm1",
        "prompt_name": "p1",
        "grade_level": 10,
    }
    r = client.post("/grade", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    # response keys and types
    assert "score" in data and isinstance(data["score"], int)
    assert 1 <= data["score"] <= 6
    assert "feedback" in data and isinstance(data["feedback"], str)
    assert "confidence" in data
    assert 0.0 <= float(data["confidence"]) <= 1.0


def test_grade_reject_large_payload():
    # Create very large text
    payload = {
        "essay_id": "e200",
        "full_text": "x" * 50000,  # exceeds default limit 20000
        "assignment": "asm1",
    }
    r = client.post("/grade", json=payload)
    assert r.status_code == 413
