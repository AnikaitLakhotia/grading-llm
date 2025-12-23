# src/baseline/prompt_runner.py
"""PromptRunner: formats prompts, calls LLMClient, and returns structured outputs."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional

from src.baseline.llm_client import LLMClient

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


DEFAULT_PROMPT_TEMPLATE = """You are a helpful grading assistant. Use the Holistic rubric (scores 1..6).
Return a JSON object with keys: score (int 1..6), feedback (short string), evidence (short string), confidence (0.0-1.0).

Rubric anchors (short):
1: minimal/no mastery; incoherent or pervasive errors.
2: little mastery; weak argument, serious coherence problems.
3: developing; inconsistent support and clarity.
4: adequate; competent reasoning with lapses.
5: strong; consistent mastery with minor lapses.
6: mastery; insightful, well-supported, nearly error-free.

--- Prompt ---
{prompt_context}

Essay:
{essay}

Respond with JSON only.
"""


class PromptRunner:
    """
    Formats prompts and queries an LLMClient.

    Example:
        runner = PromptRunner(llm_client=MockLLMClient())
        out = runner.run_single(essay_text, prompt_context="Prompt text")
    """

    def __init__(self, llm_client: LLMClient, template: Optional[str] = None):
        self.llm = llm_client
        self.template = template or DEFAULT_PROMPT_TEMPLATE

    def _format_prompt(self, essay: str, prompt_context: str = "") -> str:
        return self.template.format(essay=essay, prompt_context=prompt_context)

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """
        Parse LLM raw output into structured dict with keys:
        - score (int)
        - feedback (str)
        - evidence (str)
        - confidence (float)
        """
        # Many LLMs return plain text; our Mock returns JSON string. Try JSON parse then fallback.
        try:
            parsed = json.loads(raw)
            # validate keys
            score = int(parsed.get("score"))
            feedback = str(parsed.get("feedback", ""))
            evidence = str(parsed.get("evidence", ""))
            confidence = float(parsed.get("confidence", 0.0))
            return {
                "score": score,
                "feedback": feedback,
                "evidence": evidence,
                "confidence": confidence,
            }
        except Exception:
            # fallback: try to extract digits for score and heuristics
            # naive fallback: find first digit 1..6
            import re

            m = re.search(r"[1-6]", raw)
            score = int(m.group(0)) if m else 1
            return {
                "score": score,
                "feedback": raw[:200],
                "evidence": "",
                "confidence": 0.5,
            }

    def run_single(self, essay: str, prompt_context: str = "") -> Dict[str, Any]:
        """Run the prompt for a single essay and return structured output."""
        prompt = self._format_prompt(essay, prompt_context)
        raw = self.llm.generate(prompt)
        out = self._parse_response(raw)
        return out

    def run_batch(
        self, essays: Iterable[str], prompt_contexts: Optional[Iterable[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run on a batch of essays. prompt_contexts may be None or iterable of same length.
        Returns list of structured outputs.
        """
        if prompt_contexts is None:
            prompt_contexts = [""] * len(list(essays))
        outputs: List[Dict[str, Any]] = []
        for essay, ctx in zip(essays, prompt_contexts):
            outputs.append(self.run_single(essay=essay, prompt_context=ctx))
        return outputs
