"""LangGraph wiring: nodes, conditional routing, and human-in-the-loop.

Workflow :

    property_agent -> market_agent -> [retry gate] -> rag_agent
      -> investment_metrics_agent -> risk_assessment_agent
      -> recommendation_agent -> guardrail_agent
      -> [refuse -> final_report] | [request_reanalysis -> recommendation_agent]
      -> human_review -> [approved -> final_report] | [rejected -> recommendation_agent]
      -> final_report -> END

Human approval is REQUIRED before every final report (the only bypass is an
outright guardrail refusal, where there is nothing valid to approve).
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from property_advisor.agents.guardrail_agent import guardrail_agent
from property_advisor.agents.market_agent import market_agent
from property_advisor.agents.metrics_agent import investment_metrics_agent
from property_advisor.agents.property_agent import property_agent
from property_advisor.agents.rag_agent import rag_agent
from property_advisor.agents.recommendation_agent import recommendation_agent
from property_advisor.agents.risk_agent import risk_assessment_agent
from property_advisor.config import MAX_DATA_RETRIES
from property_advisor.logging_utils import timed_node, traced_router
from property_advisor.state import PropertyState


def data_retry_increment_node(state: PropertyState) -> dict:
    return {"data_retry_count": state.data_retry_count + 1}


@traced_router("route_after_market")
def route_after_market(state: PropertyState) -> Literal["retry", "continue"]:
    data_missing = not state.property_data or not state.market_data
    if data_missing and state.data_retry_count < MAX_DATA_RETRIES:
        return "retry"
    return "continue"


@traced_router("route_after_guardrail")
def route_after_guardrail(state: PropertyState) -> Literal["reanalyze", "refuse", "human_review"]:
    status = state.guardrail_result.get("status", "human_review_required")
    if status == "request_reanalysis":
        return "reanalyze"
    if status == "refuse":
        return "refuse"
    return "human_review"


@timed_node("human_review")
def human_review_node(state: PropertyState) -> dict:
    payload = {
        "property_address": state.property_address,
        "recommendation": state.recommendation,
        "investment_metrics": state.investment_metrics,
        "risk_assessment": {k: v for k, v in state.risk_assessment.items() if k != "raw_indicators"},
        "guardrail_result": state.guardrail_result,
    }
    decision = interrupt(payload)
    return {"human_decision": decision}


@traced_router("route_after_human")
def route_after_human(state: PropertyState) -> Literal["approved", "rejected"]:
    return "approved" if state.human_decision.get("approved") else "rejected"


@timed_node("final_report")
def final_report_node(state: PropertyState) -> dict:
    if state.guardrail_result.get("status") == "refuse":
        report = {
            "status": "refused",
            "property_address": state.property_address,
            "reason": "Guardrail Agent refused this recommendation: no valid decision/justification was produced.",
            "guardrail_result": state.guardrail_result,
        }
        return {"final_report": report, "workflow_status": "refused"}

    report = {
        "status": "approved",
        "property_address": state.property_address,
        "budget_inr": state.budget,
        "investment_horizon_years": state.investment_horizon_years,
        "investment_strategy": state.investment_strategy,
        "property_data": state.property_data,
        "market_data": state.market_data,
        "recommendation": state.recommendation,
        "investment_metrics": state.investment_metrics,
        "risk_assessment": {k: v for k, v in state.risk_assessment.items() if k != "raw_indicators"},
        "guardrail_result": state.guardrail_result,
        "human_decision": state.human_decision,
        "evidence_sources": [r["source"] for r in state.rag_context],
    }
    return {"final_report": report, "workflow_status": "completed"}


def build_graph():
    graph = StateGraph(PropertyState)

    graph.add_node("property_agent", property_agent)
    graph.add_node("market_agent", market_agent)
    graph.add_node("data_retry_increment", data_retry_increment_node)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("investment_metrics_agent", investment_metrics_agent)
    graph.add_node("risk_assessment_agent", risk_assessment_agent)
    graph.add_node("recommendation_agent", recommendation_agent)
    graph.add_node("guardrail_agent", guardrail_agent)
    graph.add_node("human_review", human_review_node)
    graph.add_node("final_report", final_report_node)

    graph.add_edge(START, "property_agent")
    graph.add_edge("property_agent", "market_agent")
    graph.add_conditional_edges(
        "market_agent",
        route_after_market,
        {"retry": "data_retry_increment", "continue": "rag_agent"},
    )
    graph.add_edge("data_retry_increment", "property_agent")
    graph.add_edge("rag_agent", "investment_metrics_agent")
    graph.add_edge("investment_metrics_agent", "risk_assessment_agent")
    graph.add_edge("risk_assessment_agent", "recommendation_agent")
    graph.add_edge("recommendation_agent", "guardrail_agent")
    graph.add_conditional_edges(
        "guardrail_agent",
        route_after_guardrail,
        {"reanalyze": "recommendation_agent", "refuse": "final_report", "human_review": "human_review"},
    )
    graph.add_conditional_edges(
        "human_review",
        route_after_human,
        {"approved": "final_report", "rejected": "recommendation_agent"},
    )
    graph.add_edge("final_report", END)

    return graph.compile(checkpointer=MemorySaver())
