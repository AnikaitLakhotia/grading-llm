# src/api/main.py
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status

try:
    # Prefer pydantic (works with v1 and with some v2 installs if compatibility shim present)
    from pydantic import BaseSettings  # type: ignore
except Exception:
    try:
        # pydantic v2+ separate settings package
        from pydantic_settings import BaseSettings  # type: ignore
    except Exception:
        # Fallback: provide a very small BaseSettings-like shim that reads env vars.
        class BaseSettings:  # pragma: no cover - fallback for very old environments
            def __init__(self, **kwargs: Any):
                # allow attribute access for any provided defaults via keyword args
                for k, v in kwargs.items():
                    setattr(self, k, v)

            @classmethod
            def __get_validators__(cls):
                if False:
                    yield


# Now the rest of the imports
from src.api.schemas import GradeRequest, GradeResponse
from src.baseline.llm_client import MockLLMClient
from src.baseline.prompt_runner import PromptRunner


# Config via env (using BaseSettings if available)
class Settings(BaseSettings):
    APP_NAME: str = os.environ.get("APP_NAME", "grading-llm-api")
    APP_VERSION: str = os.environ.get("APP_VERSION", "0.0.0")
    APP_ENV: str = os.environ.get("APP_ENV", "dev")
    USE_MOCK_LLM: bool = bool(os.environ.get("USE_MOCK_LLM", "1") == "1")
    MAX_FULL_TEXT_CHARS: int = int(os.environ.get("MAX_FULL_TEXT_CHARS", str(20_000)))
    LOG_JSON: bool = bool(os.environ.get("LOG_JSON", "1") == "1")


settings = Settings()

app = FastAPI(
    title=getattr(settings, "APP_NAME", "grading-llm-api"),
    version=getattr(settings, "APP_VERSION", "0.0.0"),
)


def _get_git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return sha.decode("utf-8").strip()
    except Exception:
        return "unknown"


APP_GIT_SHA = os.environ.get("APP_GIT_SHA", _get_git_sha())


def _json_log(entry: dict):
    if getattr(settings, "LOG_JSON", True):
        print(json.dumps(entry, default=str), flush=True)
    else:
        print(entry)


# Dependency to provide an LLM client / PromptRunner
def get_prompt_runner() -> PromptRunner:
    if getattr(settings, "USE_MOCK_LLM", True):
        llm = MockLLMClient()
    else:
        # Placeholder for real LLM client wiring
        llm = MockLLMClient()
    return PromptRunner(llm_client=llm)


@app.get("/health", response_model=dict)
async def health():
    payload = {
        "status": "ok",
        "version": getattr(settings, "APP_VERSION", "0.0.0"),
        "git_sha": APP_GIT_SHA,
        "env": getattr(settings, "APP_ENV", "local"),
    }
    return payload


@app.post("/grade", response_model=GradeResponse)
async def grade(
    req: GradeRequest,
    request: Request,
    prompt_runner: PromptRunner = Depends(get_prompt_runner),
):
    if len(req.full_text) > getattr(settings, "MAX_FULL_TEXT_CHARS", 20000):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"full_text too large (max {getattr(settings, 'MAX_FULL_TEXT_CHARS', 20000)} chars)",
        )

    req_id = f"{int(time.time() * 1000)}"
    start = time.time()
    try:
        prompt_context = f"Assignment: {req.assignment or 'unknown'} | Prompt: {req.prompt_name or 'unknown'} | Grade level: {req.grade_level or 'unknown'}"
        out = prompt_runner.run_single(
            essay=req.full_text, prompt_context=prompt_context
        )
        score = int(out.get("score", 1))
        score = max(1, min(6, score))
        feedback = str(out.get("feedback", ""))[:1000]
        evidence = str(out.get("evidence", ""))[:1000]
        confidence = float(out.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        res = GradeResponse(
            score=score, feedback=feedback, evidence=evidence, confidence=confidence
        )
        latency_ms = int((time.time() - start) * 1000)
        _json_log(
            {
                "event": "grade_request",
                "request_id": req_id,
                "essay_id": req.essay_id,
                "score": res.score,
                "confidence": res.confidence,
                "latency_ms": latency_ms,
            }
        )
        return res
    except Exception as exc:
        latency_ms = int((time.time() - start) * 1000)
        _json_log(
            {
                "event": "grade_error",
                "request_id": req_id,
                "error": str(exc),
                "latency_ms": latency_ms,
            }
        )
        raise HTTPException(status_code=500, detail="Internal grading error")
