<div align="center">

# WealthWise

**Personal asset-allocation advisory multi-agent** — Supervisor + 5-expert group on LangGraph · China investor-suitability guardrails (C1–C5) · A/HK/US market coverage · Dual cross-check (multi-source consensus + multi-model jury) · Three-layer guardrails · Langfuse trace · SSE workbench · Reproducible eval gate

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)
[![Langfuse](https://img.shields.io/badge/Langfuse-tracing-fbbf24.svg)](https://langfuse.com/)
[![CI](https://img.shields.io/badge/CI-tests%20%2B%20eval%20gate-2088FF.svg?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![eval suite](https://img.shields.io/badge/eval%20suite-53%2F53-brightgreen.svg)](#eval)
[![suitability leaks](https://img.shields.io/badge/suitability%20leaks-0-brightgreen.svg)](#eval)
[![injection block](https://img.shields.io/badge/injection%20block-100%25-brightgreen.svg)](#eval)

「武道AI / AI Engineering Dojo」**以阵制胜** series · 阵 03 · personal advisory multi-agent closed loop

</div>

---

> **EDUCATIONAL DEMO DISCLAIMER**
>
> WealthWise is an **educational demo** built for the 武道AI 阵03 case study series. It is
> **NOT investment advice**, **NOT a real compliance system**, and **NOT suitable for
> real advisory decisions**. The investor-suitability rules, risk-level mapping, and
> compliance corpus are **synthetic and paraphrased** — they do not represent the actual
> regulatory requirements of CSRC, AMAC, or any other authority. The portfolio
> optimizer uses toy risk estimates, not real covariance data. Do not use this software
> to make real financial decisions.

---

WealthWise is a personal asset-allocation advisory pipeline. Give it an investor
profile (risk level C1–C5, goals, horizon, FX consent), and it routes through a
Supervisor + 5-expert LangGraph pipeline to produce a portfolio allocation with
suitability verdict and full disclosures. The pipeline runs **fully offline and
deterministically** by default — no keys, no network. Switch `USE_REAL_PROVIDERS=true`
to wire in GLM + DeepSeek-V3 jury and AkShare live market data.

## Architecture

```text
InvestorProfile
      │
      ▼
intake (normalize / validate)
      │
      ▼
input_guard ── injection / PII screen → GUARDRAIL_BLOCKED
      │
      ▼
planner ── goal_constraints: horizon bucket, FX ceiling, R-level ceiling
      │
      ▼
budget_macro ── estimate jury calls; BUDGET_EXCEEDED → END
      │
      ▼
macro ── RAG macro-context (CPI, PMI, yield curve) + jury
      │
      ▼
equity ── screen equity candidates (A/HK/US, R-level filter)
      │
      ▼
cap ── process guardrail: dedupe + candidate cap
      │
      ▼
portfolio ── risk-parity / equal-weight optimizer   ◄──────┐
      │                                                     │
      ▼                                              DOWNGRADE + retry
budget_compliance ── estimate jury calls; BUDGET_EXCEEDED → END
      │
      ▼
compliance ── suitability check + multi-model jury (GLM × DeepSeek-V3)
      │
      ▼
reflection ── PASS → explanation
              DOWNGRADE (budget allows) → portfolio (single retry)
              DOWNGRADE (exhausted) / REJECT → finalize
      │
      ▼
explanation ── deterministic advisory text with disclosures
      │
      ▼
output_guard ── enforce disclosure completeness (FX, cross-border, risk-level)
      │
      ▼
DONE   (full trace returned; Langfuse synced if enabled)
```

**Five expert agents** (each a node in the LangGraph):

| Expert | Role |
| --- | --- |
| Goal | Parses investor goals and horizon into `goal_constraints` (R-level ceiling, FX ceiling, liquidity floor) |
| Macro | Retrieves macro context via RAG (CPI, PMI, yield-curve snippet), feeds jury for a macro outlook label |
| Equity | Screens A/HK/US equity and fixed-income candidates by R-level eligibility |
| Portfolio | Risk-parity / equal-weight optimizer; produces `PortfolioAllocation` with `fx_exposure` and `portfolio_r_level` |
| Compliance | China C1–C5 suitability check + multi-model jury (GLM primary + DeepSeek-V3 cross-check); produces PASS / DOWNGRADE / REJECT |

**Dual cross-check:**

- *Multi-source consensus* (pillar 1) — macro signals from multiple AkShare tables are
  resolved to a median with confidence capped at 0.5 for single-source readings.
- *Multi-model jury* (pillar 2) — compliance and macro verdicts go to a two-model
  panel (GLM + DeepSeek-V3, different labs for independence); majority label wins;
  low-confidence cases are flagged for human review.

**Three-layer guardrails:**

1. *Input* — profile text normalized + injection-classified (role-hijack / prompt-override / zero-width smuggling) → `GUARDRAIL_BLOCKED`.
2. *Process* — candidate deduplication, R-level ceiling enforcement, FX-exposure cap.
3. *Output* — disclosure completeness check; FX + cross-border risk disclosures mandatory when holding HK/US assets.

## Tech Stack

| Concern | Choice |
| --- | --- |
| API / orchestration | FastAPI · LangGraph |
| Primary model | **GLM-4.7** (z.ai, OpenAI-compatible) |
| Cross-check model | **DeepSeek-V3** (SiliconFlow — cross-lab independent judge) |
| Real market data | **AkShare** (A/HK/US equity, fixed income, macro) |
| Offline runtime | Sample provider + local hash embedding + in-memory vector store + offline jury (zero keys, zero network) |
| RAG | In-memory cosine store + local hash embedding; retrieves macro context and compliance corpus |
| Observability | **Langfuse** v4 (optional) + internal trace tree returned with every response |
| Eval / CI | pytest + multi-suite hard-gate CLI + GitHub Actions |
| Persistence | run/audit: memory default · sqlite (`RUN_STORE=sqlite`) · Postgres/pgvector reserved |
| Deployment | Dockerfile + docker-compose (offline out of the box) |

## Quick Start

**Prerequisites:** Python 3.12+. The default runtime is **completely offline and
deterministic** — no keys required.

```bash
# 1. Clone and create a virtual environment
git clone <repo-url> wealthwise && cd wealthwise
python3 -m venv .venv

# 2. Install (base + LLM extras; data extra adds AkShare)
.venv/bin/pip install -e '.[dev,llm]'

# 3. Run tests (offline, hermetic)
make test           # → 346+ passed

# 4. Run eval gate (offline, hard gates)
make eval           # → 53/53, suitability leaks 0, injection block 100%

# 5. Start the workbench
make run            # uvicorn wealthwise.app:app --reload
```

Open <http://localhost:8000/workbench> and submit an investor profile.

**Switching to real providers (keyed path):**

```bash
cp .env.example .env
# Fill in GLM_API_KEY, SILICONFLOW_API_KEY, and optionally LANGFUSE_* keys
# Set USE_REAL_PROVIDERS=true

make run   # now uses GLM + DeepSeek-V3 jury + AkShare live data
```

See [`docs/real-data-verification.md`](docs/real-data-verification.md) for the full
keyed-path verification guide, including AkShare column-name calibration steps.

**Docker (offline, no keys needed):**

```bash
docker compose up        # builds image, starts on :8000
# open http://localhost:8000/workbench
```

**Langfuse smoke check (keyed path only):**

```bash
make langfuse-check
# → sent wealthwise.langfuse_smoke: wealthwise
# With no keys → prints "tracing disabled / no keys — skipping" and exits 0
```

## Usage Example

```bash
# Full advisory run (JSON response)
curl -X POST localhost:8000/workbench/run \
  -H 'Content-Type: application/json' \
  -d '{
    "risk_level": "C3",
    "investable": 500000,
    "horizon_years": 5,
    "goals": ["balanced_growth"],
    "liquidity_min": 0.2,
    "accept_cross_border": true,
    "holdings": []
  }'
```

```jsonc
{
  "summary": {
    "status": "done",
    "portfolio_r_level": "R3",
    "compliance_decision": "PASS",
    "fx_exposure": 0.15,
    "budget_spent": 4
  },
  "portfolio": {
    "weights": { "600519.SH": 0.30, "2800.HK": 0.15, "000001.SH": 0.35, "CASH": 0.20 },
    "class_weights": { "equity": 0.65, "cash": 0.20, "bond": 0.15 }
  },
  "compliance": {
    "decision": "PASS",
    "disclosures": ["汇率风险：持仓含港股/美股，存在汇率波动风险。", "..."],
    "confidence": 0.9
  },
  "trace": [
    { "node": "intake", "status": "ok" },
    { "node": "macro", "status": "ok", "macro_label": "neutral" },
    "..."
  ]
}
```

```bash
# Node-by-node SSE stream
curl -N 'localhost:8000/workbench/stream?risk_level=C3&investable=500000&horizon_years=5&goals=balanced_growth&liquidity_min=0.2&accept_cross_border=true'
```

Injection attacks on `goals` or profile text fields hit the input guardrail and return
`status: GUARDRAIL_BLOCKED` immediately.

## Eval

```bash
make eval   # = python -m wealthwise.eval  (all 5 suites)
```

**5 suites / 53 cases** (fully offline, hermetic):

| Suite | Coverage |
| --- | --- |
| `golden` | End-to-end pipeline: happy paths across C1–C5, FX consent on/off, budget termination, guardrail block |
| `suitability` | Direct suitability function: over-level violations, liquidity breach, cross-border without consent — hard gate: 0 leaks |
| `misleading` | Misleading-language detection: guaranteed-return / risk-free claims must be blocked 100%; clean disclaimers must not be flagged |
| `cross_border` | Cross-border disclosure completeness + unauthorized-holding gate: FX disclosure required when holding HK/US assets |
| `robustness` | Injection block rate, invariance (formatting variants must not change compliance decision), benign false-positive rate |

**Latest baseline:**

| Metric | Value | Gate |
| --- | --- | --- |
| total_cases | **53** | >= 30 |
| pass_rate | **1.000** | 1.0 |
| decision_accuracy | **1.000** | >= 0.8 |
| **suitability_leaks** | **0** | = 0 (exit 2) |
| misleading_block_rate | **1.000** | = 1.0 (exit 2) |
| **cross_border_leaks** | **0** | = 0 (exit 2) |
| injection_block_rate | **1.000** | >= 1.0 (exit 2) |
| invariance_pass_rate | **1.000** | >= 1.0 (exit 2) |
| false_positive_rate | **0.000** | = 0.0 (exit 2) |
| **Gate** | | **PASS** |

> Eval suites live in [`data/evals/`](data/evals/). Report is written to
> [`data/evals/report.md`](data/evals/report.md) on each run.

## Observability

Every advisory run returns a `trace` list of node events (`node`, `status`,
`budget_spent`, etc.). With Langfuse configured, the same boundaries are
emitted as external observations:

```
intake → input_guard → planner → budget_macro → macro → equity
  → cap → portfolio → budget_compliance → compliance → reflection
  → explanation → output_guard
```

Tests use a conftest autouse fixture that disables tracing, ensuring the
suite is hermetic and never emits to a live Langfuse project.

## Configuration

`.env` (see [`.env.example`](.env.example)) key fields:

| Variable | Description |
| --- | --- |
| `USE_REAL_PROVIDERS` | `true` to enable AkShare data + real LLM jury (default: `false`) |
| `GLM_API_KEY` / `GLM_BASE_URL` | Primary LLM (GLM-4.7, z.ai OpenAI-compatible gateway) |
| `SILICONFLOW_API_KEY` | Cross-check jury model key (SiliconFlow) |
| `CROSSCHECK_MODEL` | Cross-check model name (default: `deepseek-ai/DeepSeek-V3`) |
| `ENABLE_LANGFUSE_TRACING` | `true` to emit traces to Langfuse |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse project credentials |
| `LANGFUSE_BASE_URL` | Langfuse endpoint (default: `https://us.cloud.langfuse.com`) |
| `MAX_FX_EXPOSURE` | Max non-CNY fraction of portfolio (default: `0.5`) |
| `MAX_LLM_JUDGMENTS` | Hard budget cap on LLM calls per run (default: `12`) |
| `RISK_BUDGET_METHOD` | Portfolio optimizer method: `risk_parity` / `equal_weight` / `mean_variance` |
| `RUN_STORE` | Audit persistence: `memory` (default) or `sqlite` |
| `TOKEN_PRICE_PER_1K` | Blended $/1k tokens for cost accounting |

## Project Structure

```text
wealthwise/
├── README.md
├── pyproject.toml              # deps + wealthwise-eval / -langfuse-check entry points
├── Makefile                    # make install / test / eval / langfuse-check / run
├── Dockerfile                  # python:3.12-slim, offline out of the box
├── docker-compose.yml          # compose service + optional Postgres stub
├── .env.example
├── docs/
│   └── real-data-verification.md  # keyed-path verification + AkShare calibration guide
├── data/
│   ├── samples/                # offline provider corpus
│   │   ├── equity.json         # A/HK/US equity candidates
│   │   ├── fixedincome.json    # bond / fixed-income candidates
│   │   ├── macro.json          # macro context snippets (CPI, PMI, yield)
│   │   └── compliance.json     # compliance corpus fragments (RAG)
│   └── evals/                  # 5 suites / 53 cases
│       ├── golden.json
│       ├── suitability.json
│       ├── misleading.json
│       ├── cross_border.json
│       ├── robustness.json
│       └── report.md
├── src/wealthwise/
│   ├── app.py                  # FastAPI: /health · /workbench(page/run/stream) · /runs
│   ├── bootstrap.py            # build_sample_deps / build_runtime_deps
│   ├── config.py               # pydantic-settings (all env vars)
│   ├── crosscheck/             # multi-model jury: deliberate(), offline stub
│   ├── eval.py                 # multi-suite hard-gate CLI
│   ├── langfuse_check.py       # Langfuse smoke check (offline-safe)
│   ├── llm.py                  # Verdict / ModelClient / OpenAI-compat client
│   ├── obs.py                  # Langfuse env bridge + traced() decorator
│   ├── runner.py               # run_advisory: build deps + invoke graph + validate
│   ├── store.py                # RunStore: memory / sqlite backends
│   ├── workbench.py            # SSE streaming + dashboard aggregation
│   ├── agents/
│   │   ├── state.py            # InvestorProfile / AssetCandidate / AdvisoryState / …
│   │   ├── deps.py             # AdvisoryDeps (providers + thresholds + budget)
│   │   ├── experts/            # goal / macro / equity / portfolio / compliance nodes
│   │   └── supervisor/         # graph.py (LangGraph) · planner.py
│   ├── compliance/             # suitability.py · language.py (misleading detection)
│   ├── guardrails/             # input.py · output.py · process.py
│   ├── portfolio/              # optimizer (risk-parity / equal-weight)
│   ├── providers/              # SampleProvider · AkShareProvider (Protocol)
│   ├── rag/                    # embed (local hash) · store (in-memory cosine) · corpus
│   └── security/               # sanitize.py (injection detection) · redact.py (PII)
└── tests/                      # pytest (346+ passed, offline hermetic)
```

## Roadmap / Honest Gaps

Delivered:

- [x] Supervisor + 5-expert LangGraph pipeline (goal / macro / equity / portfolio / compliance)
- [x] Multi-source consensus (pillar 1) + multi-model jury (pillar 2)
- [x] Three-layer guardrails (input / process / output) + budget guardrail
- [x] China investor-suitability (C1–C5) + cross-border FX rules + misleading-language detection
- [x] A/HK/US market coverage (sample provider + AkShare real-provider skeleton)
- [x] RAG macro context + compliance corpus (in-memory, local hash embedding)
- [x] Langfuse full-trace observability (optional, offline-safe)
- [x] SSE workbench + run audit store (memory / sqlite)
- [x] Multi-suite eval gate with hard gates (53 cases)
- [x] Docker + docker-compose + CI (GitHub Actions)
- [x] Persistence (RunStore: memory + sqlite, Postgres reserved)

Known gaps and future work:

- **AkShare column-name calibration** — every column assumption in
  `src/wealthwise/providers/akshare_provider.py` is marked
  `TODO(live-calibration)` and requires verification against actual live
  AkShare output before production use. See
  [`docs/real-data-verification.md`](docs/real-data-verification.md).
- **Debate / hierarchical multi-agent** — the current jury is a flat majority vote;
  a structured debate loop (Supervisor mediates expert disagreements) is future work.
- **Postgres run store** — schema is reserved in `store.py`; only `memory` and
  `sqlite` backends are implemented.
- **Real covariance data** — the portfolio optimizer uses toy risk estimates from the
  sample provider. Real mean-variance optimization requires a proper covariance matrix
  from live price history.
- **Paid data sources** — Helium10-equivalent quantitative data for Chinese markets
  (e.g. Wind, iFinD) is not integrated; the AkShare provider covers public data only.

## License

[MIT](LICENSE) (c) 2026 Kevin Tu (WealthWise / 武道AI)
