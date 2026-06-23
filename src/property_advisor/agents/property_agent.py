"""Agent 1: Property Analysis Agent.

Deterministic, tool-based: parses the Indian address and retrieves property
details from the Property Data Tool. No LLM call — per CLAUDE.md's coding
standard to prefer deterministic workflows and keep business logic out of
prompts wherever a reliable tool can do the job.
"""

from __future__ import annotations

from property_advisor.logging_utils import timed_node
from property_advisor.state import PropertyState
from property_advisor.tools.property_data_tool import get_property_data


@timed_node("property_agent")
def property_agent(state: PropertyState) -> dict:
    property_data = get_property_data(state.property_address)

    errors = list(state.errors)
    if not property_data:
        errors.append(f"No property data found for address: {state.property_address!r}")

    return {"property_data": property_data, "errors": errors}
