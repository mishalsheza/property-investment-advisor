# Property AI - Indian Property Investment Advisor

A LangGraph-based property investment advisor for the Indian real estate market.
Given a property address, INR budget, horizon, and strategy, it produces a grounded
BUY / HOLD / AVOID recommendation, routes it through guardrails, and requires human
approval before generating reports.

Ships with:

- CLI: `main.py`
- FastAPI backend: `app.py`
- Static frontend: `frontend/index.html`

All entry points use the same LangGraph workflow.

## Architecture

The graph has seven agents over a shared `PropertyState`:

```mermaid
graph TD;
	__start__([start]):::first
	property_agent(1. Property Analysis Agent)
	market_agent(2. Market Trends Agent)
	data_retry_increment(retry gate)
	rag_agent(3. RAG Research Agent)
	investment_metrics_agent(4. Investment Metrics Agent)
	risk_assessment_agent(5. Risk Assessment Agent)
	recommendation_agent(6. Recommendation Agent — LLM)
	guardrail_agent(7. Guardrail Agent — checks + LLM)
	human_review(Human Approval — interrupt)
	final_report(Final Report)
	__end__([end]):::last

	__start__ --> property_agent --> market_agent
	market_agent -. data missing, retries left .-> data_retry_increment --> property_agent
	market_agent -. data ok / retries exhausted .-> rag_agent
	rag_agent --> investment_metrics_agent --> risk_assessment_agent --> recommendation_agent
	recommendation_agent --> guardrail_agent
	guardrail_agent -. unsupported claims, capped retries .-> recommendation_agent
	guardrail_agent -. no valid decision .-> final_report
	guardrail_agent -. human_review_required .-> human_review
	human_review -. approved .-> final_report
	human_review -. rejected + feedback .-> recommendation_agent
	final_report --> __end__
```

- Agents 1-5 are deterministic/tool-based: property lookup, market lookup, RAG,
  investment metrics, and risk scoring.
- The Recommendation Agent uses Groq for the judgment call and returns decision,
  justification, evidence, and confidence.
- The Guardrail Agent keeps checks concise: missing data, high risk, weak confidence,
  negative operating cash flow, conflicting RAG evidence, and unsupported claims.
- Human review is mandatory before a final report is produced.

## Low-Level Design (LLD)

The Property AI system is implemented using a modular multi-agent architecture orchestrated by **LangGraph**.

LangGraph maintains a shared `PropertyState` throughout execution, routes control between agents, manages retry logic for missing data and recommendation validation, and coordinates the mandatory human-in-the-loop approval process before report generation.

Each agent performs a single responsibility and updates only the fields relevant to its task, making the workflow modular, maintainable, and easy to extend.
```mermaid
classDiagram

%% ==========================
%% ORCHESTRATOR
%% ==========================

class LangGraph {
    +build_graph()
    +route_workflow()
    +manage_state()
    +retry_missing_data()
    +reanalyse_recommendation()
    +interrupt_for_human_review()
}

%% ==========================
%% SHARED STATE
%% ==========================

class PropertyState {
    +property_address : str
    +budget : float
    +investment_horizon_years : int
    +investment_strategy : str

    +property_data : dict
    +market_data : dict
    +rag_context : list

    +investment_metrics : dict
    +risk_assessment : dict

    +recommendation : dict
    +guardrail_result : dict

    +requires_human_review : bool
    +human_decision : dict
    +final_report : dict

    +data_retry_count : int
    +reanalysis_retry_count : int
    +workflow_status : str
    +errors : list
}

%% ==========================
%% AGENTS
%% ==========================

class PropertyAgent{
    +lookup_property()
    +validate_input()
}

class MarketAgent{
    +fetch_market_data()
}

class RAGAgent{
    +query_chromadb()
    +retrieve_documents()
    +build_rag_context()
}

class InvestmentMetricsAgent{
    +calculate_roi()
    +calculate_cap_rate()
    +calculate_rental_yield()
    +calculate_cash_flow()
}

class RiskAssessmentAgent{
    +compute_risk_score()
}

class RecommendationAgent{
    +generate_recommendation()
}

class GuardrailAgent{
    +check_risk()
    +check_confidence()
    +check_missing_data()
    +check_conflicting_evidence()
    +validate_claims()
}

class HumanApproval{
    +approve()
    +reject()
}

class ReportGenerator{
    +generate_json()
    +generate_markdown()
    +generate_pdf()
}

%% ==========================
%% DATA SOURCES
%% ==========================

class MockData{
    +mock_properties.json
    +mock_market_trends.json
    +mock_risk_data.json
}

class ChromaDB{
    +vector_search()
}

class GroqLLM{
    +invoke()
}

%% ==========================
%% ORCHESTRATION
%% ==========================

LangGraph --> PropertyAgent
LangGraph --> MarketAgent
LangGraph --> RAGAgent
LangGraph --> InvestmentMetricsAgent
LangGraph --> RiskAssessmentAgent
LangGraph --> RecommendationAgent
LangGraph --> GuardrailAgent
LangGraph --> HumanApproval

%% ==========================
%% SHARED STATE
%% ==========================

PropertyAgent --> PropertyState
MarketAgent --> PropertyState
RAGAgent --> PropertyState
InvestmentMetricsAgent --> PropertyState
RiskAssessmentAgent --> PropertyState
RecommendationAgent --> PropertyState
GuardrailAgent --> PropertyState
HumanApproval --> PropertyState
ReportGenerator --> PropertyState

%% ==========================
%% EXTERNAL DEPENDENCIES
%% ==========================

PropertyAgent --> MockData
MarketAgent --> MockData
RiskAssessmentAgent --> MockData

RAGAgent --> ChromaDB

RecommendationAgent --> GroqLLM
GuardrailAgent --> GroqLLM

%% ==========================
%% FINAL FLOW
%% ==========================

GuardrailAgent --> HumanApproval
HumanApproval --> ReportGenerator
```


## Data

No live scraping. The project uses curated mock data and a local RAG corpus:

- `src/property_advisor/data/mock_properties.json`
- `src/property_advisor/data/mock_market_trends.json`
- `src/property_advisor/data/mock_risk_data.json`
- `src/property_advisor/data/rag_corpus/*.txt`

The app runs offline except for Groq LLM calls. ChromaDB handles local vector retrieval;
no embedding API key is required.

## Installation

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Add your Groq key to `.env`:

```bash
GROQ_API_KEY=...
```

Optional:

```bash
GROQ_MODEL=llama-3.3-70b-versatile
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=...
```

## Run the CLI

```bash
.venv/bin/python main.py --address "Whitefield, Bangalore" --budget 9500000 \
  --horizon 5 --strategy rental
```

Omit flags to be prompted interactively. At the human-review step, approve or reject
the recommendation. Approval generates JSON, Markdown, and PDF reports under
`reports/`.

Useful flags:

- `--auto-approve` skips the approval prompt.
- `--no-reports` skips report generation.
- `--reports-dir <path>` changes the report destination.
- `--demo` runs fixed demo scenarios.

For manual presentation steps, see [DEMO_WALKTHROUGH.md](DEMO_WALKTHROUGH.md).

## Run the API and Frontend

```bash
.venv/bin/python app.py
```

Open `http://localhost:8080/`.

The browser flow mirrors the graph: analyze a property, review the recommendation and
guardrail reasons, then approve or reject. Approval generates downloadable reports.

Main endpoints:

- `POST /analyze`
- `POST /approve`
- `POST /reject`
- `GET /reports/{filename}`

## Reports

Generated reports include the property summary, market data, financial metrics, risk
assessment, recommendation, guardrail reasons, human decision, and timestamp.

Output files:

```text
reports/<property_slug>.json
reports/<property_slug>.md
reports/<property_slug>.pdf
```

## Tests and Evals

Deterministic tests:

```bash
.venv/bin/python -m pytest tests/ -v
```

Evaluation scenarios, requiring `GROQ_API_KEY`:

```bash
.venv/bin/python evals/run_evals.py
```

The evals cover high-growth, weak-return, flood-risk, missing-data, and conflicting
evidence scenarios.

## Observability

Structured agent and router logs are written to `logs/project.log`.

LangSmith tracing is optional. Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`
in `.env` to enable it.
