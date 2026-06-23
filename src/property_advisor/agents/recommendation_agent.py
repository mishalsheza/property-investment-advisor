"""Agent 6: Recommendation Agent.

The one place in the graph where an LLM (Groq) exercises judgment: weighing
deterministic investment metrics, risk assessment, and RAG-retrieved
evidence into a BUY/HOLD/AVOID call with justification, supporting evidence,
and a confidence score. The prompt explicitly grounds the model in the
already-computed numbers so it cannot invent its own financial figures.

Rental-strategy cash-flow safety net (see CLAUDE.md amendments): negative
cash flow must never be silently waved through into a BUY. The prompt tells
the model the rules, and a deterministic post-processing step enforces them
regardless of what the model actually outputs — financial decision rules are
not left to LLM discretion alone, consistent with this project's preference
for deterministic guardrails over agent autonomy.

Token budget: this call is intentionally terse on both sides. The prompt
sends only the fields the model needs to decide (not full nested
property/market dicts), and the schema asks for short JSON, not prose — see
config.get_llm's max_tokens/temperature defaults and schemas.py.
"""

from __future__ import annotations

import json

from property_advisor.config import get_llm
from property_advisor.logging_utils import timed_node
from property_advisor.schemas import RecommendationOutput
from property_advisor.state import PropertyState

SYSTEM_PROMPT = """Indian real-estate Recommendation Agent. Output BUY, HOLD, or AVOID.

Rules:
- Use ONLY the numbers given below. Never invent figures.
- rental strategy: weigh rental_yield_pct and cash_flow_severity most.
- flip / long_term_appreciation: weigh roi_pct and appreciation most.
- High risk_score or conflicting evidence -> lower confidence_score.

Cash-flow rules (mandatory):
- negative_cash_flow=true: never default to BUY; lower confidence_score; \
justification must name the cash-flow figure and the tradeoff.
- cash_flow_severity="significantly_negative": decision must be HOLD or AVOID \
UNLESS strong_appreciation_evidence=true (a given flag). If true, BUY is allowed \
but justification must say why appreciation outweighs the cash-flow drag.

Be concise. Max 2 sentences in justification, up to 3 supporting_evidence items."""


def _format_rag_context(rag_context: list[dict], limit: int = 3, snippet_len: int = 220) -> str:
    if not rag_context:
        return "none"
    lines = []
    for r in rag_context[:limit]:
        lines.append(f"[{r['source']}] {r['text'][:snippet_len]}")
    return "\n".join(lines)


def _compact_metrics(metrics: dict) -> dict:
    keys = (
        "roi_pct",
        "rental_yield_pct",
        "cap_rate_pct",
        "annual_cash_flow_inr",
        "cash_flow_severity",
        "negative_cash_flow",
        "strong_appreciation_evidence",
        "break_even_years",
        "data_quality_confidence",
    )
    return {k: metrics[k] for k in keys if k in metrics}


def _compact_risk(risk: dict) -> dict:
    keys = ("risk_score", "data_quality_confidence")
    return {k: risk[k] for k in keys if k in risk}


def _enforce_cashflow_safety_net(result: RecommendationOutput, state: PropertyState) -> RecommendationOutput:
    """Deterministic safety net: the LLM is instructed to follow the
    cash-flow tradeoff rules above, but financial decision rules are not
    left to LLM discretion alone — this guarantees the rules hold even if
    the model doesn't comply.
    """
    metrics = state.investment_metrics
    if not metrics or state.investment_strategy != "rental":
        return result

    negative_cash_flow = bool(metrics.get("negative_cash_flow"))
    significantly_negative = metrics.get("cash_flow_severity") == "significantly_negative"
    strong_appreciation = bool(metrics.get("strong_appreciation_evidence"))

    decision = result.decision
    confidence = result.confidence_score
    justification = result.justification
    overridden = False

    if negative_cash_flow:
        # Never automatically BUY on the back of negative cash flow alone,
        # and always reduce confidence to reflect the added uncertainty.
        confidence = round(confidence * 0.8, 2)

    if significantly_negative and not strong_appreciation and decision == "BUY":
        decision = "HOLD"
        confidence = round(min(confidence, 0.55), 2)
        overridden = True
        justification = (
            f"{justification} [Guardrail override: the Recommendation Agent's original BUY call "
            f"was downgraded to HOLD because annual_cash_flow_inr="
            f"{metrics.get('annual_cash_flow_inr')} is significantly negative for a rental "
            f"strategy and strong_appreciation_evidence was not met — appreciation alone is not "
            f"strong enough to outweigh the cash-flow drag.]"
        )

    if not overridden and negative_cash_flow and "cash flow" not in justification.lower():
        # Safety net for the explanation requirement even if the model forgot to mention it.
        justification = (
            f"{justification} [Note: annual_cash_flow_inr={metrics.get('annual_cash_flow_inr')} "
            f"is negative; this reduces the confidence of this recommendation for a rental investor.]"
        )

    return RecommendationOutput(
        decision=decision,
        justification=justification,
        supporting_evidence=result.supporting_evidence,
        confidence_score=max(0.0, min(1.0, confidence)),
    )


@timed_node("recommendation_agent")
def recommendation_agent(state: PropertyState) -> dict:
    llm = get_llm()
    structured_llm = llm.with_structured_output(RecommendationOutput)

    feedback_note = ""
    if state.human_decision.get("approved") is False:
        feedback_note = f"\nRejected by reviewer. Feedback: {state.human_decision.get('feedback', '')}"

    reanalysis_note = ""
    if state.reanalysis_retry_count > 0:
        reasons = state.guardrail_result.get("reasons", [])
        reanalysis_note = f"\nRe-analysis #{state.reanalysis_retry_count}. Guardrail reasons: {json.dumps(reasons)}"

    user_prompt = f"""Address: {state.property_address}
Budget: INR {state.budget} | Horizon: {state.investment_horizon_years}yr | Strategy: {state.investment_strategy}

Property: {json.dumps({k: state.property_data.get(k) for k in ("city", "locality", "property_type", "price_inr", "area_sqft") if k in state.property_data})}
Market: {json.dumps(state.market_data) if state.market_data else "none"}
Metrics: {json.dumps(_compact_metrics(state.investment_metrics))}
Risk: {json.dumps(_compact_risk(state.risk_assessment))}
RAG:
{_format_rag_context(state.rag_context)}
{feedback_note}{reanalysis_note}
"""

    result: RecommendationOutput = structured_llm.invoke(
        [("system", SYSTEM_PROMPT), ("user", user_prompt)]
    )
    result = _enforce_cashflow_safety_net(result, state)

    return {"recommendation": result.model_dump()}
