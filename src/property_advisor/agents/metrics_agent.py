"""Agent 4: Investment Metrics Agent.

Deterministic, tool-based ROI/cap-rate/rental-yield/cash-flow/break-even
calculations in INR. No LLM call — financial calculations must be
deterministic.
"""

from __future__ import annotations

from property_advisor.logging_utils import timed_node
from property_advisor.state import PropertyState
from property_advisor.tools.financial_calculator import compute_investment_metrics


@timed_node("investment_metrics_agent")
def investment_metrics_agent(state: PropertyState) -> dict:
    metrics = compute_investment_metrics(
        property_data=state.property_data,
        market_data=state.market_data,
        budget=state.budget,
        horizon_years=state.investment_horizon_years,
    )
    return {"investment_metrics": metrics}
