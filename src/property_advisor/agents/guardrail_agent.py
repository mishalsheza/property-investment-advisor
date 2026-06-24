"""Agent 7: Guardrail Agent.

Validates the Recommendation Agent's output before it can reach a human
reviewer. Combines cheap deterministic checks (missing data, risk score,
negative cash flow, confidence threshold, known conflicting-evidence pairs)
with one targeted LLM call to catch claims in the justification that aren't
grounded in the supplied metrics/context — the kind of check that's hard to
do reliably with regexes.

Routing outcomes (see graph.py):
- "request_reanalysis": loop back to the Recommendation Agent (capped by
  MAX_REANALYSIS_RETRIES) — used only for fixable issues (unsupported claims).
- "refuse": the recommendation is fundamentally unusable; end the graph
  without human approval (nothing valid to approve).
- "human_review_required": forward to Human Approval, which is REQUIRED for
  every run per CLAUDE.md regardless of guardrail outcome. `reasons` lists
  every concern a human reviewer should weigh, even if it's empty (human
  approval is still mandatory with no flagged concerns).
"""

from __future__ import annotations

import json

from langsmith import traceable

from property_advisor.config import (
    GUARDRAIL_CONFIDENCE_THRESHOLD,
    MAX_REANALYSIS_RETRIES,
    RISK_HUMAN_REVIEW_THRESHOLD,
    get_llm,
)
from property_advisor.logging_utils import timed_node
from property_advisor.schemas import ClaimAudit
from property_advisor.state import PropertyState

# Known pairs of sources in the RAG corpus that intentionally present
# conflicting analyst views (see data/rag_corpus/hinjewadi_*_analyst_note.txt).
# A real deployment would detect this via semantic contradiction analysis;
# an explicit pair list keeps this check deterministic and testable here.
CONFLICT_PAIRS = [
    ("hinjewadi_bullish_analyst_note.txt", "hinjewadi_bearish_analyst_note.txt"),
]

CLAIM_AUDIT_SYSTEM_PROMPT = """Fact-check this real-estate justification against the grounded \
data. Flag ONLY claims (numbers, property/market attributes, named risks) with NO basis in the \
grounded data and not in the RAG source list.

Do NOT flag: paraphrases of numbers actually in the data (e.g. risk_score=13 called "low"), \
reasonable inferences from property/market/metrics/risk data, generic reasoning, or bracketed \
[Guardrail override: ...] / [Note: ...] text (added by this pipeline, not the model).

Be concise. Up to 3 flagged fragments max, no explanation."""


@traceable(name="guardrail_detect_conflicting_evidence", run_type="tool")
def _detect_conflicting_evidence(rag_context: list[dict]) -> list[str]:
    sources = {r["source"] for r in rag_context}
    conflicts = []
    for a, b in CONFLICT_PAIRS:
        if a in sources and b in sources:
            conflicts.append(f"{a} vs {b}")
    return conflicts


@traceable(name="guardrail_audit_claims", run_type="chain")
def _audit_claims(state: PropertyState) -> ClaimAudit:
    llm = get_llm()
    structured_llm = llm.with_structured_output(ClaimAudit)

    risk_assessment = {k: v for k, v in state.risk_assessment.items() if k != "raw_indicators"}
    grounded_data = {
        "property_data": state.property_data,
        "market_data": state.market_data,
        "investment_metrics": state.investment_metrics,
        "risk_assessment": risk_assessment,
        "rag_sources": [r["source"] for r in state.rag_context],
    }
    user_prompt = f"""Justification: {state.recommendation.get('justification', '')}
Evidence cited: {json.dumps(state.recommendation.get('supporting_evidence', []))}
Grounded data: {json.dumps(grounded_data)}"""
    return structured_llm.invoke(
        [("system", CLAIM_AUDIT_SYSTEM_PROMPT), ("user", user_prompt)]
    )


@timed_node("guardrail_agent")
def guardrail_agent(state: PropertyState) -> dict:
    reasons: list[str] = []

    missing_property_data = not state.property_data
    missing_market_data = not state.market_data
    if missing_property_data:
        reasons.append("Property data is missing or incomplete.")
    if missing_market_data:
        reasons.append("Market trend data is missing for this locality (data-quality risk).")

    risk_score = state.risk_assessment.get("risk_score", 0)
    high_risk = risk_score > RISK_HUMAN_REVIEW_THRESHOLD
    if high_risk:
        reasons.append(f"Risk score {risk_score} exceeds the human-review threshold of {RISK_HUMAN_REVIEW_THRESHOLD}.")

    negative_cash_flow = bool(state.investment_metrics.get("negative_cash_flow"))
    if negative_cash_flow:
        reasons.append(
            f"Negative annual cash flow (INR {state.investment_metrics.get('annual_cash_flow_inr')})."
        )

    confidence = state.recommendation.get("confidence_score", 0.0)
    confidence_below_threshold = confidence < GUARDRAIL_CONFIDENCE_THRESHOLD
    if confidence_below_threshold:
        reasons.append(f"Recommendation confidence {confidence:.2f} is below threshold {GUARDRAIL_CONFIDENCE_THRESHOLD}.")

    conflicts = _detect_conflicting_evidence(state.rag_context)
    conflicting_evidence = bool(conflicts)
    if conflicting_evidence:
        reasons.append(f"Conflicting evidence detected in retrieved context: {', '.join(conflicts)}.")

    claim_audit = _audit_claims(state)
    if claim_audit.has_unsupported_claims:
        reasons.append("Unsupported claims found: " + "; ".join(claim_audit.unsupported_claims))

    decision_present = bool(state.recommendation.get("decision"))
    is_refusable = not decision_present or not state.recommendation.get("justification")

    if is_refusable:
        status = "refuse"
    elif claim_audit.has_unsupported_claims and state.reanalysis_retry_count < MAX_REANALYSIS_RETRIES:
        # Only re-run the Recommendation Agent for fixable issues (claims it
        # can correct by re-prompting). Risk/cash-flow/confidence concerns
        # are facts about the deal, not reasoning errors — re-analysis won't
        # change them, so those go straight to a human reviewer instead.
        status = "request_reanalysis"
    else:
        status = "human_review_required"

    requires_human_review = (
        state.requires_human_review
        or high_risk
        or negative_cash_flow
        or confidence_below_threshold
        or missing_property_data
        or missing_market_data
        or conflicting_evidence
    )

    guardrail_result = {
        "status": status,
        "reasons": reasons,
        "missing_property_data": missing_property_data,
        "missing_market_data": missing_market_data,
        "high_risk": high_risk,
        "negative_cash_flow": negative_cash_flow,
        "confidence_below_threshold": confidence_below_threshold,
        "conflicting_evidence": conflicting_evidence,
        "conflicts": conflicts,
        "has_unsupported_claims": claim_audit.has_unsupported_claims,
        "unsupported_claims": claim_audit.unsupported_claims,
    }

    update = {
        "guardrail_result": guardrail_result,
        "requires_human_review": requires_human_review,
    }
    if status == "request_reanalysis":
        update["reanalysis_retry_count"] = state.reanalysis_retry_count + 1

    return update
