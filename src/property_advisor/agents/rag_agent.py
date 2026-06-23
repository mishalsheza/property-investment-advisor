"""Agent 3: RAG Research Agent.

Deterministic vector retrieval over the Indian real-estate corpus (reports,
zoning/RERA notes, metro expansion plans). Synthesis/grounding against this
context happens downstream in the Recommendation and Guardrail agents.
"""

from __future__ import annotations

from property_advisor.logging_utils import timed_node
from property_advisor.state import PropertyState
from property_advisor.tools.rag_tool import query_rag


@timed_node("rag_agent")
def rag_agent(state: PropertyState) -> dict:
    city = state.property_data.get("city", "")
    locality = state.property_data.get("locality", "")

    query = (
        f"{locality} {city} real estate appreciation trends, zoning and RERA "
        f"regulations, infrastructure/metro development, flood and regulatory risk, "
        f"relevant to a {state.investment_strategy} investment strategy"
    ).strip()

    results = query_rag(query, k=6)

    # de-duplicate by source while preserving order
    seen = set()
    deduped = []
    for r in results:
        if r["source"] not in seen:
            seen.add(r["source"])
            deduped.append(r)

    return {"rag_context": deduped}
