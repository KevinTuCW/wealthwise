# Real-Data Verification Guide

This document describes how to verify the WealthWise pipeline with real LLM providers
and real AkShare market data.  No API keys are bundled with the repo; you must supply
your own.

## Environment Variables to Set

Copy `.env.example` to `.env` and fill in:

```bash
# Switch to real providers (AkShare data + real model jury)
USE_REAL_PROVIDERS=true

# Primary LLM (GLM via z.ai, OpenAI-compatible)
GLM_API_KEY=<your-z.ai-key>
GLM_BASE_URL=https://api.z.ai/api/paas/v4/  # default, can omit

# Cross-check / jury model (SiliconFlow / DeepSeek-V3, cross-lab judge)
SILICONFLOW_API_KEY=<your-siliconflow-key>
SILICONFLOW_BASE_URL=https://api.siliconflow.com/v1  # default, can omit
CROSSCHECK_MODEL=deepseek-ai/DeepSeek-V3

# Langfuse observability (optional; leave blank to run without tracing)
ENABLE_LANGFUSE_TRACING=true
LANGFUSE_PUBLIC_KEY=<your-langfuse-public-key>
LANGFUSE_SECRET_KEY=<your-langfuse-secret-key>
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com  # or your self-hosted URL
```

## Running the App with Real Providers

```bash
# Option A: via Make
cp .env.example .env  # edit .env with your keys
make run              # uvicorn wealthwise.app:app --reload

# Option B: explicit env override (no .env file needed)
USE_REAL_PROVIDERS=true \
  GLM_API_KEY=<key> \
  SILICONFLOW_API_KEY=<key> \
  PYTHONPATH=src .venv/bin/python -m uvicorn wealthwise.app:app --reload
```

Open <http://localhost:8000/workbench> and submit an investor profile.

## Langfuse Smoke Check

After setting Langfuse keys:

```bash
make langfuse-check
# or directly:
PYTHONPATH=src .venv/bin/python -m wealthwise.langfuse_check
```

Expected output: `sent wealthwise.langfuse_smoke: wealthwise`

## AkShare Column-Name Calibration

The real-data AkShare provider (`src/wealthwise/providers/akshare_provider.py`) contains
numerous `TODO(live-calibration)` markers.  These mark every place where an AkShare
DataFrame column name was assumed from documentation and **must be verified against
actual live output** before production use.

### Why They Exist

AkShare is a community library whose column names change across versions and whose
documentation can lag behind the actual API.  The sample provider (used in tests and
offline mode) uses stable synthetic data; the AkShare provider makes best-effort guesses
for column names like `"收盘"`, `"wind_code"`, `"jjdm"`, etc.

### How to Calibrate

1. Set `USE_REAL_PROVIDERS=true` and ensure `akshare` is installed (`pip install -e '.[data]'`).
2. Open a Python shell and call the relevant AkShare function directly, e.g.:

   ```python
   import akshare as ak
   df = ak.stock_zh_a_spot_em()
   print(df.columns.tolist())  # compare against the expected column names in the provider
   ```

3. For each `TODO(live-calibration)` marker in `akshare_provider.py`, check the comment,
   run the corresponding `ak.*` call, and update the column names as needed.

4. Key files and functions to check:
   - `AkShareEquityProvider.get_candidates()` — A-share spot tables
   - `AkShareEquityProvider.get_macro()` — macro indicators (CPI, PMI, GDP)
   - `AkShareFixedIncomeProvider.get_candidates()` — bond fund / fixed-income lists
   - `AkShareFixedIncomeProvider.get_macro()` — bond market indicators
   - `_parse_r_level()` — R-level parsing from fund risk-level fields

5. After updating column names, re-run the full test and eval suites:

   ```bash
   make test   # 346+ cases, all offline (unchanged)
   make eval   # 53 eval cases, hard gates must pass
   ```

   The eval suites are hermetic and use the sample provider, so they are not affected
   by AkShare column changes — but any real-data path you add should be smoke-tested
   manually with live data.

## What "Live Calibration" Does NOT Affect

- All unit tests and eval suites run exclusively against the **sample provider** (offline,
  deterministic).  They are not affected by AkShare or LLM keys and will always pass
  without any keys.
- The eval hard gates (suitability 0-leak, misleading 1.0 block rate, cross-border 0-leak,
  injection 1.0 block rate, invariance 1.0 pass rate, false-positive 0.0 rate) are
  enforced on the offline suite and are **not relaxed** for real-data paths.
