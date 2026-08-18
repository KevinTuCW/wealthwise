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

## Verified Run — 2026-08-10 (real keys)

A keyed end-to-end run was executed via `scripts/verify_real.py` (sample market data +
**real GLM-4.7 + DeepSeek-V3 jury** + **real Langfuse tracing**). Results:

| Profile | status | decision | portfolio R | fx exposure | real tokens | latency |
|---|---|---|---|---|---|---|
| C2 conservative (no cross-border) | done | PASS | R2 | 0.0% | 1946 | 34.5s |
| C4 balanced (cross-border) | done | PASS | R4 | 3.96% | 1876 | 38.7s |

- **Multi-model jury is real**: the macro-tilt `deliberate()` call consumed ~1.9k tokens
  per advisory across GLM-4.7 + DeepSeek-V3. Latency (~35–39s) is dominated by GLM's
  thinking mode on the macro call. Compliance jury only fires on DOWNGRADE/REJECT or high
  FX, so PASS cases show `tokens: 0` at the compliance node (expected).
- **Cross-border FX** is computed for real (C4 held HK/US → fx_exposure 3.96%); C2
  (accept_cross_border=False) correctly stayed A-share only (fx 0.0%).
- **conservative_mode** (C1/C2) tightened equity screening (PE cap 25) as designed.
- **Langfuse**: `make langfuse-check` → `sent wealthwise.langfuse_smoke`; advisory runs
  emit generation/embedding spans via the `langfuse.openai` drop-in when tracing is on.
- Harmless noise: an `httpx SyncHttpxClientWrapper.__del__ AttributeError` prints at GC
  on Python 3.14 — cosmetic, not a functional error.

### AkShare live reachability (from this environment)

- **Reachable & column-verified**: `ak.fund_open_fund_daily_em()` → cols include
  `基金代码 / 基金简称 / {date}-单位净值 / 日增长率 / 申购状态 / 赎回状态 / 手续费`
  (matches the provider's `基金代码/基金简称` mapping); `ak.macro_china_lpr()` → cols
  `TRADE_DATE / LPR1Y / LPR5Y`.
- **Replaced**: `ak.stock_zh_a_spot_em()` (host `82.push2.eastmoney.com`). The equity path
  no longer uses it — see below.

### Why the equity provider moved off AkShare

An earlier revision of this document blamed the `stock_zh_a_spot_em()` failure on a
"host-specific egress limitation". That diagnosis was wrong on both counts, and is
corrected here because it would send the next person looking in the wrong place.

Measured from this environment:

- The host is fine. `82.push2.eastmoney.com` resolves (14.103.191.91), TCP 443 connects,
  and it serves a valid `*.eastmoney.com` certificate (DigiCert / GeoTrust, current). It
  returns real A-share data whenever a request completes. `ping` failing means nothing —
  these quote hosts drop ICMP.
- Not host-specific. `quote.eastmoney.com` and the numbered `push2` shards behave alike;
  what varies is the individual connection, not the hostname.
- A local proxy was part of it. `urllib.request.getproxies()` returned `127.0.0.1:1087`
  **even with every `*_proxy` environment variable unset**, because requests reads the
  macOS system proxy settings and not just the environment. Only `Session.trust_env = False`
  actually bypasses it. This is where the `RECORD_LAYER_FAILURE` / `bad decrypt` in the
  tracebacks came from: the TLS handshake succeeds, then the record stream is mangled.
- Bypassing the proxy is not sufficient. Direct calls succeed roughly one time in three;
  failures reset in 0.12–0.25s (fast RST, not a timeout). That residual interference is
  not fixable in code.

A source that fails two calls in three is the problem regardless of whose fault it is,
because the failure lands inside `equity_node`, which has no fallback, and takes the whole
advisory run down with it (HTTP 500 from `/workbench/run`).

The equity path therefore moved to `TencentMarketProvider` (`qt.gtimg.cn`), which answers
A + HK + US from one batched endpoint. Measured: 300 A-share + 32 HK + 40 US candidates in
**0.8s total**, against a paginated screener that mostly did not complete. Field positions
(`name=1, code=2, price=3, pe=39, mcap=45, pb=46`) are verified live on all three market
layouts, so the equity mapper no longer carries `TODO(live-calibration)`.

Because that endpoint quotes symbols rather than screening a market, `screen()` now draws
from a shipped universe (`data/universe.json`: CSI 300 + curated HK/US large caps,
regenerated by `scripts/refresh_universe.py`). Keeping the universe in the repo is
deliberate — fetching the constituent list live would reinstate exactly the kind of network
dependency this change removed.

`AkShareFundProvider` / `AkShareMacroProvider` / `AkShareFXProvider` are unchanged; their
endpoints (`fund_open_fund_daily_em`, `macro_china_lpr`, BOC FX) are reachable and verified
above.

> Note: some function/class names in the calibration section above are indicative; the
> fund / macro / FX providers use `AkShareFundProvider` / `AkShareMacroProvider` /
> `AkShareFXProvider` with a `_get` seam + `_map_*` helpers. Equities use
> `TencentMarketProvider`, same `_get` seam, backed by `data/universe.json`.

## What "Live Calibration" Does NOT Affect

- All unit tests and eval suites run exclusively against the **sample provider** (offline,
  deterministic).  They are not affected by AkShare or LLM keys and will always pass
  without any keys.
- The eval hard gates (suitability 0-leak, misleading 1.0 block rate, cross-border 0-leak,
  injection 1.0 block rate, invariance 1.0 pass rate, false-positive 0.0 rate) are
  enforced on the offline suite and are **not relaxed** for real-data paths.
