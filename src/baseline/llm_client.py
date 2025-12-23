# src/baseline/llm_client.py
"""LLM client abstraction and Mock implementation for deterministic testing."""

from __future__ import annotations

import abc
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LLMClient(abc.ABC):
    """Abstract LLM client interface."""

    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        """
        Send prompt to an LLM and return raw text output.

        Implementations should raise exceptions on failure.
        """
        raise NotImplementedError


@dataclass
class MockLLMClient(LLMClient):
    """
    Deterministic mock LLM client used for testing and CI.

    Behavior:
    - Maps input prompt deterministically to a score in 1..6 derived from SHA256 hash.
    - Returns a JSON-string that includes score, feedback and confidence fields.
    """

    salt: str = "mock-llm-salt-v1"

    def _deterministic_score(self, prompt: str) -> int:
        h = hashlib.sha256()
        h.update(self.salt.encode("utf-8"))
        h.update(prompt.encode("utf-8"))
        digest = h.digest()
        # map to 0..5 then +1 for 1..6
        val = digest[0] % 6
        return int(val + 1)

    def _deterministic_confidence(self, prompt: str) -> float:
        # derive a pseudo-random confidence 0.5..0.95
        h = hashlib.sha256()
        h.update((self.salt + "c").encode("utf-8"))
        h.update(prompt.encode("utf-8"))
        digest = h.digest()
        val = digest[1] / 255.0
        # scale to 0.5-0.95
        return float(0.5 + 0.45 * val)

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        """Return a JSON string with deterministic score, feedback, evidence, and confidence."""
        score = self._deterministic_score(prompt)
        confidence = self._deterministic_confidence(prompt)
        feedback = f"Mock feedback: predicted score {score} (based on prompt hash)."
        evidence = "Mock evidence: see highlighted phrases."
        payload = {
            "score": int(score),
            "feedback": feedback,
            "evidence": evidence,
            "confidence": float(confidence),
        }
        # return JSON string (simulates an LLM returning a JSON blob)
        return json.dumps(payload)
