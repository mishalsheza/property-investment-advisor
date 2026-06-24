# Property Alpha AI — Indian Property Investment Advisor

A multi-agent real estate investment analysis platform for the **Indian real estate
market**, built on LangGraph. Given a property address, an INR budget, an investment
horizon, and a strategy (rental / flip / long-term appreciation), it produces a
grounded BUY / HOLD / AVOID recommendation that passes through deterministic
guardrails and a mandatory human approval step before any final report is produced.

Ships as a CLI (`main.py`), a FastAPI backend (`app.py`), and a static frontend
(`frontend/index.html`) — all driving the *same* LangGraph workflow.


## Architecture

Seven agents, wired as a LangGraph `StateGraph` over a shared `PropertyState`
(Pydantic model, `src/property_advisor/state.py`). This graph is the single source of
truth — the CLI, the FastAPI endpoint, and the eval/demo harnesses all call
`build_graph()` from `src/property_advisor/graph.py`; nothing duplicates this logic.

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

| # | Agent | Type | Responsibility |
|---|-------|------|-----------------|
| 1 | Property Analysis Agent | deterministic | Parse address, look up property details (mock dataset) |
| 2 | Market Trends Agent | deterministic | City/locality appreciation, demand/supply |
| 3 | RAG Research Agent | deterministic (vector search) | Retrieve RERA/zoning/metro/analyst context from ChromaDB |
| 4 | Investment Metrics Agent | deterministic | ROI, cap rate, rental yield, cash flow, EMI, `strong_appreciation_evidence` |
| 5 | Risk Assessment Agent | deterministic | Crime/flood/vacancy/regulatory risk → 0-100 score |
| 6 | Recommendation Agent | **LLM (Groq)** + deterministic safety net | BUY/HOLD/AVOID + justification + confidence |
| 7 | Guardrail Agent | deterministic checks + **LLM (Groq)** claim audit | Decides `human_review_required` / `request_reanalysis` / `refuse` |

Agents 1-5 are intentionally deterministic and tool-based —not LLM-based. Only the Recommendation Agent (judgment call) and the
Guardrail Agent's unsupported-claims check (reading comprehension) call an LLM, and
both use **Groq only**.

### State schema (`src/property_advisor/state.py`)

```python
class PropertyState(BaseModel):
    # Inputs
    property_address: str
    budget: float                          # INR
    investment_horizon_years: int = 5
    investment_strategy: Literal["rental", "flip", "long_term_appreciation"] = "rental"

    # Agent outputs
    property_data: dict = {}
    market_data: dict = {}
    rag_context: list[dict] = []
    investment_metrics: dict = {}          # includes strong_appreciation_evidence, cash_flow_severity
    risk_assessment: dict = {}
    recommendation: dict = {}              # decision, justification, supporting_evidence, confidence_score
    guardrail_result: dict = {}            # status, reasons[], + individual check flags

    # Human-in-the-loop
    requires_human_review: bool = False
    human_decision: dict = {}              # {approved, feedback}
    final_report: dict = {}

    # Routing / control
    data_retry_count: int = 0
    reanalysis_retry_count: int = 0
    workflow_status: str = "in_progress"
    errors: list[str] = []
```

Each agent reads/writes only the fields relevant to its responsibility.

### Recommendation logic & cash-flow tradeoff rules

The Recommendation Agent decides from the **whole picture** — total return (`roi_pct`,
which already folds in 5-year appreciation and cash flow), income (`rental_yield_pct`),
risk (`risk_score`), and RAG evidence — never from a single metric. The prompt
(`agents/recommendation_agent.py`) gives lean guidance (strong total return + solid yield
+ acceptable risk → lean BUY; mixed signals → HOLD; reserve AVOID for genuinely poor
deals), and a deterministic post-processing step (`_enforce_cashflow_safety_net`)
guarantees the operating cash-flow rule holds even if the model doesn't comply.

**Operating vs. leveraged cash flow (the key distinction).** `cash_flow_severity` and
`negative_cash_flow` describe the property's **operating** (unlevered) economics — whether
the asset itself earns money after expenses, independent of financing. The fully-leveraged
shortfall (an 80%-LTV EMI that net rent rarely covers) is reported **separately** as
`levered_cash_flow_negative` and treated as a financing caveat, not a disqualifier. This
matters because Indian gross rental yields (~2–6%) never cover an 80%-LTV EMI, so a
leveraged figure is deeply negative for essentially every property — using it as the
primary signal previously labeled the entire market `significantly_negative` and biased
recommendations toward AVOID (see "Recommendation-bias fix" below).

The deterministic rules are now:

- If `levered_cash_flow_negative` is true (rental strategy): confidence is multiplied by
  0.95 and a financing caveat is appended to the justification — a nudge, not a downgrade.
- If `cash_flow_severity == "significantly_negative"` (the asset loses money *before*
  financing) and `strong_appreciation_evidence` is **false**: a BUY is forcibly downgraded
  to HOLD (confidence capped at 0.55), with an override note appended.
- `strong_appreciation_evidence` (computed deterministically in
  `tools/financial_calculator.py`) requires BOTH `appreciation_rate_5yr_pct >= 8.0` AND
  `roi_pct >= 15.0` — a high ROI alone (which leverage math can produce) isn't enough;
  the underlying market appreciation rate must itself be strong.

### Recommendation-bias fix (operating vs. leveraged cash flow)

**Symptom.** The engine returned **AVOID** for almost every property despite the expanded
datasets and RAG corpus.

**Root cause.** `cash_flow_severity` and `negative_cash_flow` in
`tools/financial_calculator.py` were computed on a **fully-leveraged** basis (80% LTV @ 9%
over 20 years). Indian gross rental yields (2–5.6% in the mock data) can never cover that
EMI, so **34/34 properties** came out `significantly_negative` with `negative_cash_flow =
true`. The Recommendation Agent's prompt then carried a hard rule mandating HOLD/AVOID on
`significantly_negative` cash flow, so nearly every property was pushed to AVOID regardless
of its ROI, appreciation, yield, or risk.

**Fix (minimal, two files).**

- `tools/financial_calculator.py`: `cash_flow_severity` and `negative_cash_flow` are now
  computed from the **unlevered operating** cash flow (financing-independent), and a new
  `levered_cash_flow_negative` field carries the financing caveat separately.
- `agents/recommendation_agent.py`: the prompt now weighs the combined picture and treats
  leveraged-negative cash flow as a confidence caveat, not a near-automatic AVOID; the
  safety net only forces a BUY→HOLD downgrade when the property is *operating*-negative.

**Validation (Groq `llama-3.3-70b-versatile`).** Representative 10-property sample, same
model before and after:

| Decision | BEFORE | AFTER |
|----------|:------:|:-----:|
| BUY      | 2      | 6     |
| HOLD     | 0      | 2     |
| AVOID    | 8      | 2     |

Decisions are now discriminating, not merely flipped: strong deals (ROI 13–34%) → BUY;
genuinely poor deals (Worli ROI −9.1%, Andheri −0.96%, low yield) → still AVOID; borderline
(Dwarka ROI 2.4%) and high-risk (Dadar, flood risk_score 80) → HOLD. Confidence recovered
from a 0.16–0.64 range to 0.57–0.80. Across the full 34-property dataset the deterministic
change moves `cash_flow_severity` from 34/34 `significantly_negative` to 34/34 `positive`
(operating basis), with the leverage caveat preserved on all 34 via
`levered_cash_flow_negative`. Reproduce with `python scripts/validate_recommendations.py`.

### Guardrail checks

The Guardrail Agent (`agents/guardrail_agent.py`) evaluates every recommendation against:

- `risk_score > RISK_HUMAN_REVIEW_THRESHOLD` (default 75) — a "high" flood-risk
  classification is treated as a hard floor (risk_score >= 80) in the Risk Assessment
  Agent, since a single severe physical risk shouldn't be diluted by an otherwise-average
  weighted score.
- `negative_cash_flow == true` (operating/unlevered basis — the asset loses money before
  financing; the routine leveraged-EMI shortfall is `levered_cash_flow_negative` and does
  not by itself trigger review)
- `confidence_score < GUARDRAIL_CONFIDENCE_THRESHOLD` (default 0.70)
- `property_data` / `market_data` incomplete
- Conflicting evidence in the RAG context (explicit known-conflict source pairs)
- Unsupported claims in the justification (LLM-audited against the grounded data)

Output shape:

```json
{
  "status": "human_review_required",
  "reasons": [
    "Risk score 80.0 exceeds the human-review threshold of 75."
  ],
  "missing_property_data": false,
  "missing_market_data": false,
  "high_risk": true,
  "negative_cash_flow": false,
  "confidence_below_threshold": false,
  "conflicting_evidence": false,
  "conflicts": [],
  "has_unsupported_claims": false,
  "unsupported_claims": []
}
```

(`negative_cash_flow` is `false` for typical properties post-fix because it now reflects
operating economics; it flips to `true` only for assets that lose money before financing.)

`status` is one of:
- **`human_review_required`** — the normal path; human approval is mandatory for every
  run regardless, so this status (and its `reasons`) exists to tell the reviewer *why*
  they should look closely, not just that they must look.
- **`request_reanalysis`** — only for fixable issues (unsupported claims found by the
  LLM audit); loops back to the Recommendation Agent, capped by `MAX_REANALYSIS_RETRIES`.
  Risk/cash-flow/confidence concerns are facts about the deal, not reasoning errors, so
  they go straight to a human reviewer instead of wasting a re-analysis loop.
- **`refuse`** — the recommendation has no valid decision/justification at all; ends the
  graph without human approval (there's nothing valid to approve).

### Human-in-the-loop

Human approval is mandatory before every final report (the only bypass is an outright
guardrail refusal). Implemented with LangGraph's `interrupt()` / `Command(resume=...)`,
checkpointed with `MemorySaver`. Rejecting loops back to the Recommendation Agent with
the reviewer's feedback. The FastAPI `/analyze` endpoint auto-approves internally so it
can return a complete result synchronously to the browser — the full interactive
approve/reject loop is the CLI's primary surface (see `main.py`).

### Report generation

After approval, `src/property_advisor/report_generator.py` (ReportLab + plain
markdown/JSON) writes:

```
reports/
├── <property_slug>.json
├── <property_slug>.md
└── <property_slug>.pdf
```

Each includes: title, property address, budget, strategy, horizon, property summary,
market summary, financial metrics, risk assessment, recommendation, supporting
evidence, guardrail reasons, human approval decision, and a timestamp.

### Data sources

No live scraping — all data is a curated **mock dataset** , so the project runs offline except for Groq LLM calls:

- `src/property_advisor/data/mock_properties.json` — 5 Indian properties spanning a
  high-growth IT corridor, an expensive low-yield Tier-1 micro-market, a monsoon
  flood-risk zone, a Tier-3 city, and a corridor with conflicting analyst outlooks.
- `src/property_advisor/data/mock_market_trends.json`, `mock_risk_data.json`
- `src/property_advisor/data/rag_corpus/*.txt` — synthetic illustrative
  Knight Frank/JLL-style outlook notes, an RERA Act summary, zoning/FSI rules, metro
  expansion tracker, a Mumbai monsoon flood-risk note, and two intentionally
  *conflicting* Hinjewadi analyst notes (used to exercise the Guardrail Agent's
  conflicting-evidence check). These are clearly labeled as synthetic/illustrative,
  not real published figures.

RAG retrieval uses ChromaDB with its built-in local embedding model (no embeddings API
key required — Groq has no embeddings endpoint, and the project is Groq-only for LLM
calls).

## Installation

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY
```

## Environment variables (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | _(required)_ | The only supported LLM provider |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model id |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing (legacy alias: `LANGCHAIN_TRACING_V2`) |
| `LANGSMITH_API_KEY` | _(empty)_ | LangSmith key (legacy alias: `LANGCHAIN_API_KEY`); tracing no-ops gracefully if unset even when the flag above is `true` |
| `LANGSMITH_PROJECT` | `property-investment-advisor` | LangSmith project name (legacy alias: `LANGCHAIN_PROJECT`) |
| `RISK_HUMAN_REVIEW_THRESHOLD` | `75` | Risk score above which the Risk/Guardrail agents flag human review |
| `GUARDRAIL_CONFIDENCE_THRESHOLD` | `0.70` | Confidence below which the Guardrail Agent flags human review |
| `MAX_DATA_RETRIES` | `2` | Cap on the missing-data retry loop |
| `MAX_REANALYSIS_RETRIES` | `2` | Cap on the guardrail re-analysis loop |
| `AUTO_APPROVE_HUMAN_REVIEW` | `false` | CLI default for `--auto-approve` |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | FastAPI bind address |

## Running — CLI

```bash
.venv/bin/python main.py --address "Whitefield, Bangalore" --budget 9500000 \
    --horizon 5 --strategy rental
```

Omit flags to be prompted interactively. You'll be shown the recommendation, risk
assessment, and guardrail result (with `reasons`), and asked to approve or reject (with
feedback) at the human-review step. After approval, JSON/Markdown/PDF reports are
generated automatically under `reports/` (use `--no-reports` to skip, `--reports-dir` to
change the destination). Use `--auto-approve` to skip the interactive prompt.

## Running — Demo mode

```bash
.venv/bin/python main.py --demo
```

Runs 3 fixed, auto-approved scenarios back-to-back and generates a full report set for
each:

1. **High-growth Bangalore property** (Whitefield) → expect BUY
2. **Negative cash flow property** (Worli, Mumbai) → expect AVOID
3. **High risk (flood-prone) property** (Dadar, Mumbai) → expect AVOID + human-review flag

Console output ends with, per case:

```
✓ Recommendation approved
✓ JSON report generated
✓ Markdown report generated
✓ PDF report generated

Report saved to:
  reports/<property_slug>.pdf
```

## Running — FastAPI backend

```bash
.venv/bin/python app.py
# or: .venv/bin/uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

This serves the frontend at `http://localhost:8000/` and mirrors the CLI's real
human-in-the-loop flow over HTTP — analysis pauses for approval, it doesn't
auto-approve. Same graph, same `MemorySaver` checkpointer, same `report_generator.py`
the CLI uses; sessions live in-process and don't survive a server restart (fine for
local/demo use). CORS is open (`*`) for local testing only.

```
POST /analyze   {"address": "...", "budget": 9500000, "horizon": 5, "strategy": "rental"}
                -> {"approval_required": true, "thread_id": "...", "recommendation": {...},
                    "investment_metrics": {...}, "risk_assessment": {...}, "guardrail_result": {...}}
                   (or {"approval_required": false, "final_report": {...}} if the Guardrail
                    Agent refused outright — nothing to approve in that case)

POST /approve   {"thread_id": "..."}
                -> resumes the graph as approved, generates reports/<slug>.{json,md,pdf}
                   via the same generate_reports() the CLI calls, returns
                   {"final_report": {...}, "report_paths": {"json": "/reports/...",
                    "markdown": "/reports/...", "pdf": "/reports/..."}}

POST /reject    {"thread_id": "...", "feedback": "..."}
                -> resumes the graph as rejected with feedback, which loops back through
                   the Recommendation Agent; returns the same shape as /analyze (a new
                   pending recommendation to approve/reject, or a final_report if refused)

GET  /reports/{filename}  -> serves a generated report file (mounted via StaticFiles)
```

```bash
curl -s -X POST http://localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"address":"Worli, Mumbai","budget":45000000,"horizon":5,"strategy":"rental"}'
# -> {"approval_required": true, "thread_id": "api-...", ...}

curl -s -X POST http://localhost:8000/approve -H "Content-Type: application/json" \
  -d '{"thread_id":"api-..."}'
# -> {"final_report": {...}, "report_paths": {"pdf": "/reports/worli_mumbai.pdf", ...}}
```

If Groq fails, endpoints return `503 {"detail": "AI recommendation service temporarily
unavailable."}` rather than a raw stack trace; a report-generation failure after a
successful approval returns `{"final_report": {...}, "report_error": "..."}` instead of
crashing.

## Running — Frontend

With `app.py` running, open `http://localhost:8000/` in a browser — the frontend is
served directly from there, so no separate step is needed.

The page is a single static file (`frontend/index.html`, vanilla HTML/CSS/JS, no
frameworks/build step). Flow: fill in the form and click **Analyze Property** → the
recommendation, risk assessment, metrics, and guardrail reasons appear in a decision
card (**green = BUY**, **yellow = HOLD**, **red = AVOID**) with **Approve** / **Reject**
buttons — nothing is final yet. Clicking **Reject** prompts for feedback and re-runs the
Recommendation Agent, showing the updated call for another approve/reject round.
Clicking **Approve** generates the reports and replaces the buttons with **Download PDF
/ JSON / Markdown Report** links.

## Demo walkthrough (manual, exact steps)

Useful when presenting live and you want to narrate each part of the spec rather than
running `--demo`. Each one exercises a different code path — read "what to look for"
before approving.

**1. Happy path — BUY on strong total return, with a leverage caveat**

```bash
.venv/bin/python main.py --address "Whitefield, Bangalore, 560066" --budget 9500000 \
  --horizon 5 --strategy rental
```
What to look for: `investment_metrics.negative_cash_flow == false` (operating cash flow is
positive) but `levered_cash_flow_negative == true`; `roi_pct` strong (~28%) and
`strong_appreciation_evidence == true`; `recommendation.decision == "BUY"` with a
justification that notes the EMI/leverage caveat; `recommendation.confidence_score` healthy
(~0.8, trimmed only ~0.95x for the caveat).

**2. AVOID case — weak total return, not merely negative leveraged cash flow**

```bash
.venv/bin/python main.py --address "Worli, Mumbai" --budget 45000000 \
  --horizon 5 --strategy rental --auto-approve
```
What to look for: `strong_appreciation_evidence == false`, `roi_pct < 0` (negative total
return), low `rental_yield_pct` (~1.6%); `recommendation.decision == "AVOID"` — driven by
the genuinely poor combined economics, not by the leverage caveat alone.

**3. Human review triggered by flood risk**

```bash
.venv/bin/python main.py --address "Dadar, Mumbai" --budget 18000000 --horizon 5 --strategy rental
```
What to look for: `risk_assessment.risk_score >= 80` (flood-risk hard floor),
`guardrail_result.reasons` includes the risk-score line.

**4. Missing-data retry workflow (Tier-3 city)**

```bash
.venv/bin/python main.py --address "Muzaffarpur, Bihar" --budget 3500000 \
  --horizon 5 --strategy long_term_appreciation --auto-approve
```
What to look for in `logs/project.log`: `property_agent` and `market_agent` running more
than once (the `market_agent -> data_retry_increment -> property_agent` loop), capped at
`MAX_DATA_RETRIES`. Final `guardrail_result.missing_market_data == true`.

**5. Conflicting analyst reports → guardrail intervention**

```bash
.venv/bin/python main.py --address "Hinjewadi Phase 2, Pune" --budget 7200000 \
  --horizon 5 --strategy rental --auto-approve
```
What to look for: `guardrail_result.conflicting_evidence == true` and `conflicts` naming
the bullish-vs-bearish Hinjewadi notes.

**6. Testing the rejection / re-analysis loop**

Run any of the above *without* `--auto-approve`, and when prompted answer `n` with
feedback — the graph loops back to the Recommendation Agent with your feedback injected
into its prompt, and re-prompts you for approval.

**7. Watch the structured logs live**

```bash
tail -f logs/project.log
```

## Tests and evaluation

Deterministic tool unit tests (no API key required):

```bash
.venv/bin/python -m pytest tests/ -v
```

The 5 required evaluation scenarios (calls Groq — needs `GROQ_API_KEY`; subject to your
Groq tier's rate limits if run repeatedly in a short window):

```bash
.venv/bin/python evals/run_evals.py
```

1. High-growth metro (Bangalore/Whitefield) → **BUY**
2. Negative cash flow, Tier-1 city (Mumbai/Worli) → **AVOID**
3. Flood-prone area (Mumbai/Dadar-Kurla) → **human review triggered**
4. Missing market data, Tier-3 city (Muzaffarpur) → **retry workflow exercised**
5. Conflicting analyst reports (Pune/Hinjewadi) → **guardrail intervention**

## Observability

Every agent and routing decision logs a structured JSON line to `logs/project.log`
(`agent_name`/`router_name`, `input_summary`, `output_summary`, `execution_time_seconds`)
and mirrors to the console — see `src/property_advisor/logging_utils.py`. Summaries are
one level deep (large nested values collapse to `{...N keys}` / `[...N items]`) so logs
stay scannable.

Every agent, both LLM calls, and all three routing functions are also wrapped with
LangSmith's `@traceable`, so each shows up as its own named span when tracing is
enabled — not just whatever LangGraph's own runnable tracing captures by default. Set
`LANGSMITH_TRACING=true` plus `LANGSMITH_API_KEY` in `.env` to enable; tracing is a
graceful no-op (never a crash) if the key is absent even with the flag set.

## Project layout

```
app.py                          # FastAPI backend
main.py                         # CLI (single run + --demo)
frontend/index.html             # Static frontend (vanilla HTML/CSS/JS)
src/property_advisor/
  state.py                      # PropertyState (shared graph state)
  config.py                     # env-driven config, Groq client, LangSmith resolution
  graph.py                      # StateGraph wiring, routing, human-in-the-loop
  logging_utils.py              # structured logging + LangSmith tracing decorators
  report_generator.py           # JSON/Markdown/PDF report generation (ReportLab)
  schemas.py                    # structured-output schemas (RecommendationOutput, ClaimAudit)
  agents/                       # the 7 agents
  tools/                        # deterministic tools (property/market/risk/RAG/financial calc)
  data/                         # mock datasets + RAG corpus
evals/                          # 5 required evaluation scenarios
tests/                          # deterministic unit tests
reports/                        # generated reports (gitignored)
logs/project.log                # structured run log (gitignored)
```

## Preserving the architecture

This refactor adds the report generator, FastAPI backend, frontend, demo mode, and
tightens the recommendation/guardrail logic — it does not remove or replace LangGraph,
the multi-agent workflow, the mandatory human approval step, RAG, guardrails, or the
structured `PropertyState` schema. All new entry points (`app.py`, `--demo`) call the
exact same `build_graph()`.
