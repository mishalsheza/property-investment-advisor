"""Shared LangGraph state schema for the Property Investment Advisor.

Every agent reads/writes only the fields relevant to its responsibility. Routing/control fields are kept separate
from the domain data fields so agents never need to touch them directly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PropertyState(BaseModel):
    # --- Inputs ---
    property_address: str
    budget: float  # INR
    investment_horizon_years: int = 5
    investment_strategy: Literal["rental", "flip", "long_term_appreciation"] = "rental"

    # --- Agent outputs ---
    property_data: dict[str, Any] = Field(default_factory=dict)
    market_data: dict[str, Any] = Field(default_factory=dict)
    rag_context: list[dict[str, Any]] = Field(default_factory=list)
    investment_metrics: dict[str, Any] = Field(default_factory=dict)
    risk_assessment: dict[str, Any] = Field(default_factory=dict)
    recommendation: dict[str, Any] = Field(default_factory=dict)
    guardrail_result: dict[str, Any] = Field(default_factory=dict)

    # --- Human-in-the-loop ---
    requires_human_review: bool = False
    human_decision: dict[str, Any] = Field(default_factory=dict)
    final_report: dict[str, Any] = Field(default_factory=dict)

    # --- Routing / control ---
    data_retry_count: int = 0
    reanalysis_retry_count: int = 0
    workflow_status: str = "in_progress"
    errors: list[str] = Field(default_factory=list)
