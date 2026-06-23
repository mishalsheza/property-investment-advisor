"""Structured-output schemas for LLM-backed agents (Recommendation, Guardrail).

Field descriptions explicitly ask for brevity — these are JSON outputs, not
prose, and every word here is an output token the model will spend.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RecommendationOutput(BaseModel):
    decision: Literal["BUY", "HOLD", "AVOID"]
    justification: str = Field(description="Max 2 short sentences. No filler.")
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Up to 3 short citations, e.g. 'roi_pct=28.7' or a filename.",
    )
    confidence_score: float = Field(ge=0.0, le=1.0)


class ClaimAudit(BaseModel):
    has_unsupported_claims: bool
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Up to 3 short claim fragments, no explanation."
    )
