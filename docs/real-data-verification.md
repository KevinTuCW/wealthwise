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

## AkShare Column-Name Calibration — done, 2026-09-01

Every column mapping in `src/wealthwise/providers/akshare_provider.py` has been checked
against live output from akshare 1.18.83, and the `TODO(live-calibration)` markers are
gone. This section records what the check found, because three of the mappings were
wrong in a way that reading the documentation would never have shown: all three returned
a plausible number.

| endpoint | columns (live) | verdict |
|---|---|---|
| `macro_china_lpr` | `TRADE_DATE / LPR1Y / LPR5Y / RATE_1 / RATE_2`, 1575 rows **oldest-first** | mapping was correct; latest print 2026-08-20, LPR1Y = 3.00 |
| `macro_china_cpi_yearly` | `商品 / 日期 / 今值 / 预测值 / 前值`, oldest-first | column right, **row wrong** — the table ends with the *next scheduled* release, 今值 = NaN, so the snapshot published a NaN CPI between prints |
| `macro_china_cpi` | `月份 / 全国-当月 / 全国-同比增长 / …`, 223 rows **newest-first** | column right, **order assumed** — reading the last row returned the January **2008** print (7.08%) instead of July 2026 (0.5%) |
| `currency_boc_sina` | `日期 / 中行汇买价 / 中行钞买价 / 中行钞卖价·汇卖价 / 央行中间价 / 中行折算价`, per **100** units | column and divisor right, **call wrong** — `start_date`/`end_date` default to constants frozen at `20230304`–`20231110`, so a no-argument call quoted a 2023 rate as today's |

Fixes: `_newest()` picks the most recent row that actually carries a reading, by date
rather than by position, which covers both the ordering and the scheduled-NaN cases; the
FX call passes an explicit recent window; and both macro and FX refuse a print that is
too old instead of returning it.

### The freshness guard, and why it costs a CPI publisher

`macro_china_cpi_yearly` is **not currently publishing**: its last real print is
2025-08-09, a year behind the statistics bureau, and every other jin10-backed series in
this akshare version (`macro_usa_cpi_monthly`, `macro_china_cpi_monthly`, the euro-area
tables) stops within days of the same date. The mapping is correct and the source now
returns nothing.

That matters more than a missing number, because it is the failure a consensus layer is
least able to catch. Two publishers a year apart do not look like a disagreement: 0.0%
from August 2025 and 0.5% from July 2026 are both plausible CPI prints, so the resolver
reconciles them into a narrow spread and reports high confidence in a figure that
describes no month at all. With the guard in place CPI falls back to one publisher at
confidence 0.5 — visibly, in the consensus record.

Live snapshot after calibration:

```text
akshare-lpr        -> {'interest_rate': 0.03}
akshare-cpi-yearly -> {}                        # stale feed, correctly silent
akshare-cpi-nbs    -> {'cpi': 0.005}
USDCNH = 6.7809    HKDCNH = 0.8650              # was 7.1771 / 0.9193 (2023-11-10)
```

### What was deleted rather than calibrated

`AkShareMarketProvider`, `AkShareFundProvider` and the `build_provider` factory are gone.
They wrapped the eastmoney spot endpoints (`stock_zh_a_spot_em` and its HK/US siblings),
which are unreachable from here — TLS handshake completes, then the stream is reset — and
nothing had referenced them since the equity path moved to Tencent. An uncallable code
path cannot be calibrated, and an uncalibrated fallback is a liability rather than a
safety net, so the honest close-out was removal, not another `TODO`.

### Re-verifying

```bash
PYTHONPATH=src .venv/bin/python -c "
from wealthwise.providers.akshare_provider import build_macro_sources, AkShareFXProvider
for s in build_macro_sources(): print(s.name, s.snapshot())
print(AkShareFXProvider().rate('USDCNH'))"

make test   # 520 offline cases — unaffected by akshare, by design
make eval   # 64 eval cases, hard gates
```

The test and eval suites are hermetic and run against the sample provider, so they do not
cover any of the above. That is the point of this document: the live mappings have no
automated gate, so they get a dated record instead.

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

- **Reachable and calibrated**: `macro_china_lpr`, `macro_china_cpi`, `currency_boc_sina`
  — column by column, above.
- **Reachable but stopped**: `macro_china_cpi_yearly`, last print 2025-08-09.
- **Unreachable**: `stock_zh_a_spot_em` and its HK/US siblings (hosts
  `82.push2.eastmoney.com` / `72.push2.eastmoney.com`). The equity path no longer uses
  them — see below.

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

What is left of `akshare_provider.py` after that move is the macro sources
(`AkShareLprSource` / `AkShareCpiYearlySource` / `AkShareCpiNbsSource`, consumed through
`build_macro_sources()`) and `AkShareFXProvider`, all behind the same `_table()` / `_get()`
seam so tests stub them without importing akshare.

One more live finding from the same session, on the Tencent side rather than AkShare: the
k-line endpoint needs the venue suffix for US names. `usAAPL` answers with a two-bar stub
whatever bar count is requested; only `usAAPL.OQ` returns the series. The quote mapper had
been stripping that suffix, so **every US name in the book was silently getting no
momentum and no realized volatility** — history is an enrichment, and a name that gets
none is simply scored on fewer factors. The suffix now survives into
`metrics["venue_code"]` and the history provider uses it.

## What "Live Calibration" Does NOT Affect

- All unit tests and eval suites run exclusively against the **sample provider** (offline,
  deterministic).  They are not affected by AkShare or LLM keys and will always pass
  without any keys.
- The eval hard gates (suitability 0-leak, misleading 1.0 block rate, cross-border 0-leak,
  injection 1.0 block rate, invariance 1.0 pass rate, false-positive 0.0 rate) are
  enforced on the offline suite and are **not relaxed** for real-data paths.
