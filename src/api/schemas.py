# src/api/schemas.py
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, validator


class GradeRequest(BaseModel):
    essay_id: str = Field(..., description="Unique essay identifier")
    full_text: str = Field(..., description="Sanitized essay text (plain UTF-8)")
    assignment: Optional[str] = Field(None, description="Assignment id or short name")
    prompt_name: Optional[str] = Field(None, description="Prompt identifier")
    grade_level: Optional[int] = Field(None, description="Grade level (e.g., 10)")

    @validator("essay_id")
    def essay_id_not_empty(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("essay_id must be a non-empty string")
        return v

    @validator("full_text")
    def full_text_min_length(cls, v: str) -> str:
        if not isinstance(v, str) or len(v.strip()) < 5:
            raise ValueError("full_text must be a non-empty string (>=5 characters)")
        return v


class GradeResponse(BaseModel):
    score: int = Field(..., ge=1, le=6, description="Predicted holistic score (1..6)")
    feedback: str = Field(..., description="Short feedback string")
    evidence: Optional[str] = Field(
        None, description="Evidence snippet or justification"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence in [0,1]"
    )

    @validator("feedback")
    def feedback_nonempty(cls, v: str) -> str:
        if v is None:
            return ""
        return v
