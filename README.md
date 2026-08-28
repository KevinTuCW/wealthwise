<div align="center">

# WealthWise

**个人资产配置投顾多智能体** — LangGraph 上的 Supervisor + 5 专家小组 · 中国投资者适当性护栏（C1–C5）· A/港/美股覆盖 · 双重交叉验证（多源共识 + 多模型陪审）· 三层护栏 · Langfuse 全链路 trace · SSE 工作台 · 可复现评测门禁

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#许可)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)
[![Langfuse](https://img.shields.io/badge/Langfuse-tracing-fbbf24.svg)](https://langfuse.com/)
[![CI](https://img.shields.io/badge/CI-tests%20%2B%20eval%20gate-2088FF.svg?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![eval suite](https://img.shields.io/badge/eval%20suite-64%2F64%20(offline)-brightgreen.svg)](#评测)
[![suitability leaks](https://img.shields.io/badge/suitability%20leaks-0-brightgreen.svg)](#评测)
[![injection block](https://img.shields.io/badge/injection%20block-100%25-brightgreen.svg)](#评测)

「武道AI / AI Engineering Dojo」**以阵制胜** 系列 · 阵 03 · 个人投顾多智能体闭环

**中文** · [English](README.en.md)

</div>

---

> **教育演示声明**
>
> WealthWise 是为「武道AI 阵03」实战案例系列构建的**教育演示项目**。它**不构成投资建议**、**不是真实合规系统**、**不适用于任何真实投顾决策**。其中的投资者适当性规则、风险等级映射与合规语料均为**合成/改写**内容，不代表中国证监会、中国证券投资基金业协会或任何监管机构的实际要求。组合优化器使用的是玩具级风险估计，而非真实协方差数据。请勿用本软件做任何真实理财决策。

---

WealthWise 是一条个人资产配置投顾流水线。给它一份投资者画像（风险等级 C1–C5、目标、期限、是否接受跨境），它会经过一条 Supervisor + 5 专家的 LangGraph 流水线，产出一份带适当性裁决与完整披露的资产配置方案。默认运行时**完全离线、确定性**——无需 key、无需网络。设 `USE_REAL_PROVIDERS=true` 即接入 GLM + DeepSeek-V3 陪审与 AkShare 实时行情。

## 架构

```text
InvestorProfile 投资者画像
      │
      ▼
intake（归一化 / 校验）
      │
      ▼
input_guard ── 注入 / PII 筛查 → GUARDRAIL_BLOCKED
      │
      ▼
planner ── goal_constraints：期限分档、外币敞口上限、R 级上限
      │
      ▼
budget_macro ── 估算陪审调用数；超预算 → BUDGET_EXCEEDED → END
      │
      ▼
macro ── RAG 宏观上下文（CPI/PMI/利率）+ 多模型陪审出大类 tilt
      │
      ▼
equity ── 筛选权益候选（A/港/美股，R 级过滤，跨境门）
      │
      ▼
cap ── 进程护栏：去重 + 候选截断
      │
      ▼
portfolio ── 反波动率 / 风险预算优化器                    ◄──────┐
      │                                                        │
      ▼                                                 DOWNGRADE + 单次重试
budget_compliance ── 估算陪审调用数；超预算 → BUDGET_EXCEEDED → END
      │
      ▼
compliance ── 适当性检查 + 多模型陪审（GLM × DeepSeek-V3，陪审只能加严）
      │
      ▼
reflection ── PASS → explanation
              DOWNGRADE（预算允许）→ portfolio（至多一次重试）
              DOWNGRADE（已耗尽）/ REJECT → finalize
      │
      ▼
explanation ── 从结构化披露组装人读方案文本
      │
      ▼
output_guard ── 校验披露完整性（适当性 / 风险 / 免责 / 跨境汇率）
      │
      ▼
DONE   （返回完整 trace；开启后同步 Langfuse）
```

**五个专家 Agent**（每个是 LangGraph 里的一个节点）：

| 专家 | 职责 |
| --- | --- |
| 目标规划 Goal | 把投资者目标与期限解析成 `goal_constraints`（R 级上限、外币敞口上限、流动性下限） |
| 宏观 Macro | 经 RAG 检索宏观上下文（CPI/PMI/利率片段），交陪审出大类资产 tilt |
| 权益 Equity | 按 R 级适配在 A/港/美股筛选权益与固收候选（不接受跨境则仅 A 股） |
| 风险组合 Portfolio | **两级**风险预算优化器：先按目标定大类中枢（`min_equity`/`max_equity`/`liquidity_min`），再在类内做反波动率（含波动下限与单一标的占比上限）；产出 `PortfolioAllocation` |
| 合规 Compliance | 中国 C1–C5 适当性检查 + 多模型陪审（GLM 主 + DeepSeek-V3 副）；产出 PASS / DOWNGRADE / REJECT，**陪审只能加严、永不软化 REJECT** |

**双重交叉验证：**

- *多源共识*（支柱一）—— 宏观/量化信号从多个 AkShare 接口汇成中位数，单源读数置信封顶 0.5。
- *多模型陪审*（支柱二）—— 合规与宏观判定交给**跨三家实验室的奇数陪审团**（GLM-4.7 智谱 / DeepSeek-V3 / Ling-flash-2.0 蚂蚁）；多数标签胜出（3/3=1.0、2/3≈0.667、三方分歧无多数记 `None`），低置信升级人工复核。奇数才有「多数」可言——两个模型只有「一致」与「平票」两种结果。**PASS 也要复核**：确定性规则只会错在「本该拦却放行」这一侧，所以 PASS 结果按 `jury_review_pass_rate` 抽样复检（默认全查），陪审依旧只能加严。

**三层护栏：**

1. *输入* —— 画像文本归一化 + 注入分类（角色劫持 / 指令改写 / 零宽走私）→ `GUARDRAIL_BLOCKED`。
2. *过程* —— 候选去重、R 级上限强制、外币敞口上限。
3. *输出* —— 披露完整性校验；持有港/美股时强制含**跨境汇率**风险披露（校验实质「汇率」措辞，而非关键词凑数）。

## 技术栈

| 关注点 | 选择 |
| --- | --- |
| API / 编排 | FastAPI · LangGraph |
| 主模型 | **GLM-4.7**（z.ai，OpenAI 兼容） |
| 交叉验证模型 | **DeepSeek-V3** + **Ling-flash-2.0**（均走 SiliconFlow）—— 与 GLM 合成跨三家实验室的奇数陪审团 |
| 真实行情 | **AkShare**（A/港/美股、基金、宏观、汇率） |
| 离线运行时 | 样例 Provider + 本地哈希嵌入 + 内存向量库 + 离线陪审（零 key、零网络） |
| RAG | 内存余弦向量库 + 本地哈希嵌入；检索宏观上下文与合规/研报语料 |
| 可观测 | **Langfuse** v4（可选）+ 每次响应返回内部 trace 树 |
| 评测 / CI | pytest + 多套件硬门 CLI + GitHub Actions |
| 持久化 | run/审计：内存默认 · sqlite（`RUN_STORE=sqlite`）· Postgres/pgvector 预留 |
| 部署 | Dockerfile + docker-compose（离线即跑） |

## 快速开始

**前置要求：** Python 3.12+。默认运行时**完全离线、确定性**——无需任何 key。

```bash
# 1. 克隆并建虚拟环境
git clone <repo-url> wealthwise && cd wealthwise
python3 -m venv .venv

# 2. 安装（base + LLM extras；data extra 额外装 AkShare）
.venv/bin/pip install -e '.[dev,llm]'

# 3. 跑测试（离线、hermetic）
make test           # → 370 passed

# 4. 跑评测门禁（离线、硬门）
make eval           # → 64/64（离线档），适当性漏判 0，配置合理性 100%，注入拦截 100%

# 5. 启动工作台
make run            # uvicorn wealthwise.app:app --reload
```

浏览器打开 <http://localhost:8000/workbench>，提交一份投资者画像。

**切换到真实 Provider（keyed 路径）：**

```bash
cp .env.example .env
# 填入 GLM_API_KEY、SILICONFLOW_API_KEY，以及可选的 LANGFUSE_* keys
# 设 USE_REAL_PROVIDERS=true

make run   # 此时使用 GLM + DeepSeek-V3 陪审 + AkShare 实时数据
```

完整的 keyed 路径验证指南（含 AkShare 列名校准步骤）见 [`docs/real-data-verification.md`](docs/real-data-verification.md)。

**Docker（离线，无需 key）：**

```bash
docker compose up        # 构建镜像，启动在 :8000
# 打开 http://localhost:8000/workbench
```

**Langfuse 连通性自检（仅 keyed 路径）：**

```bash
make langfuse-check
# → sent wealthwise.langfuse_smoke: wealthwise
# 无 key 时 → 打印 "tracing disabled / no keys — skipping" 并 exit 0
```

## 用法示例

```bash
# 完整投顾运行（JSON 响应）
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
    "disclosures": ["跨境汇率风险：持仓含港股/美股，存在汇率波动、通道与税收风险。", "..."],
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
# 逐节点 SSE 流式
curl -N 'localhost:8000/workbench/stream?risk_level=C3&investable=500000&horizon_years=5&goals=balanced_growth&liquidity_min=0.2&accept_cross_border=true'
```

对 `goals` 或画像文本字段的注入攻击会命中输入护栏，立即返回 `status: GUARDRAIL_BLOCKED`。

## 评测

```bash
make eval   # = python -m wealthwise.eval  （全部 6 套件）
```

**7 套件 / 64 例**（完全离线、hermetic）：

> 这套门禁跑在离线桩陪审上：它度量的是**规则与流水线**是否自洽，不是模型判断力。真模型的表现要用 keyed 真实验证单独跑（见 `docs/real-data-verification.md`）。

| 套件 | 覆盖 |
| --- | --- |
| `golden` | 端到端流水线：C1–C5 各档 happy path、是否接受跨境、决策与组合风险等级对齐标注 |
| `suitability` | 适当性函数直测：越级违规、流动性击穿、未授权跨境 —— 硬门：0 漏判 |
| `misleading` | 误导用语检测：保本 / 稳赚 / 承诺收益必须 100% 拦截；带免责的正常文案不得误报 |
| `cross_border` | 跨境披露完整性 + 未授权持仓门：持港/美股必须含实质汇率披露 |
| `robustness` | 注入拦截率、画像变形不变性（格式变体不得改变决策）、良性边界误报率 |
| `status_routing` | 端到端状态路由：注入越级/跨境/去流动性，验证 DOWNGRADE→需复核、REJECT→不可出具 |
| `allocation_sanity` | **配置合理性（硬门）**：权益上下限、流动性下限真达成、单一持仓上限、持仓数上限 —— 适当性的另一侧：方案不能「安全到答非所问」 |

**最新基线：**

| 指标 | 值 | 门禁 |
| --- | --- | --- |
| total_cases | **64** | >= 30 |
| pass_rate | **1.000** | 1.0 |
| decision_accuracy | **1.000** | >= 0.8 |
| **suitability_leaks** | **0** | = 0（exit 2） |
| misleading_block_rate | **1.000** | = 1.0（exit 2） |
| **cross_border_leaks** | **0** | = 0（exit 2） |
| injection_block_rate | **1.000** | >= 1.0（exit 2） |
| invariance_pass_rate | **1.000** | >= 1.0（exit 2） |
| false_positive_rate | **0.000** | = 0.0（exit 2） |
| **status_routing_accuracy** | **1.000** | >= 1.0（exit 2） |
| **allocation_sanity_rate** | **1.000** | >= 1.0（exit 2） |
| **门禁** | | **PASS** |

> 评测套件位于 [`data/evals/`](data/evals/)。每次运行生成报告 [`data/evals/report.md`](data/evals/report.md)。

## 可观测

每次投顾运行都返回一个 `trace` 节点事件列表（`node`、`status`、`budget_spent` 等）。配置 Langfuse 后，同一批边界会作为外部 observation 发出：

```
intake → input_guard → planner → budget_macro → macro → equity
  → cap → portfolio → budget_compliance → compliance → reflection
  → explanation → output_guard
```

测试通过 conftest 的 autouse fixture 强制关闭 tracing，保证测试套件 hermetic、绝不误发到线上 Langfuse 项目。

## 配置

`.env`（见 [`.env.example`](.env.example)）关键字段：

| 变量 | 说明 |
| --- | --- |
| `USE_REAL_PROVIDERS` | `true` 启用 AkShare 数据 + 真实 LLM 陪审（默认 `false`） |
| `GLM_API_KEY` / `GLM_BASE_URL` | 主模型（GLM-4.7，z.ai OpenAI 兼容网关） |
| `SILICONFLOW_API_KEY` | 交叉验证陪审模型 key（SiliconFlow） |
| `CROSSCHECK_MODEL` | 第二陪审员（默认 `deepseek-ai/DeepSeek-V3`） |
| `THIRD_MODEL` | 第三陪审员（默认 `inclusionAI/Ling-flash-2.0`，走 SiliconFlow，凑成奇数陪审团）；设空串回退两模型档 |
| `ENABLE_LANGFUSE_TRACING` | `true` 向 Langfuse 发送 trace |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse 项目凭证 |
| `LANGFUSE_BASE_URL` | Langfuse 端点（默认 `https://us.cloud.langfuse.com`） |
| `MAX_FX_EXPOSURE` | 组合非 CNY 资产最大占比（默认 `0.5`） |
| `MAX_LLM_JUDGMENTS` | 单次运行 LLM 调用硬预算上限（默认 `12`）。预算估算按**实际陪审员数**推算（一次合议 = 每位陪审员一次调用），三名陪审员的一次完整咨询实测占用 6/12，留有余量 |
| `RISK_BUDGET_METHOD` | 组合优化方法：`risk_parity`（默认；其他方法预留，未实现即 `NotImplementedError`） |
| `RUN_STORE` | 审计持久化：`memory`（默认）或 `sqlite` |
| `TOKEN_PRICE_PER_1K` | 成本核算用的混合 $/1k tokens 单价 |

## 项目结构

```text
wealthwise/
├── README.md / README.en.md    # 中文（默认）/ English
├── pyproject.toml              # 依赖 + wealthwise-eval / -langfuse-check 入口
├── Makefile                    # make install / test / eval / langfuse-check / run
├── Dockerfile                  # python:3.12-slim，离线即跑
├── docker-compose.yml          # compose 服务 + 可选 Postgres stub
├── .env.example
├── scripts/
│   └── verify_real.py          # keyed 真实数据验证脚本
├── docs/
│   └── real-data-verification.md  # keyed 路径验证 + AkShare 校准指南
├── data/
│   ├── samples/                # 离线 Provider 语料
│   │   ├── equities.json       # A/港/美股权益候选
│   │   ├── funds.json          # 基金 / 固收 / 货币候选
│   │   ├── macro.json          # 宏观上下文（CPI/PMI/利率）
│   │   ├── fx.json             # 汇率
│   │   ├── policy.json         # 适当性/风险揭示政策语料（RAG）
│   │   └── research.json       # 标的研报/资讯语料（RAG）
│   └── evals/                  # 7 套件 / 64 例
│       ├── golden.json  suitability.json  misleading.json
│       ├── cross_border.json  robustness.json  status_routing.json
│       └── report.md
├── src/wealthwise/
│   ├── app.py                  # FastAPI：/health · /workbench(page/run/stream) · /runs
│   ├── bootstrap.py            # build_sample_deps / build_runtime_deps + 离线陪审
│   ├── config.py               # pydantic-settings（全部环境变量）
│   ├── crosscheck/             # 多模型陪审：deliberate()、离线 stub
│   ├── eval.py                 # 多套件硬门 CLI
│   ├── langfuse_check.py       # Langfuse 连通性自检（离线安全）
│   ├── llm.py                  # Verdict / ModelClient / OpenAI 兼容客户端
│   ├── obs.py                  # Langfuse 环境桥接 + traced() 装饰器
│   ├── runner.py               # run_advisory：装依赖 + invoke 图 + 校验
│   ├── store.py                # RunStore：memory / sqlite 后端
│   ├── workbench.py            # SSE 流式 + 五块 Dashboard 聚合
│   ├── agents/
│   │   ├── state.py            # InvestorProfile / AssetCandidate / AdvisoryState / …
│   │   ├── deps.py             # AdvisoryDeps（Provider + 阈值 + 预算）
│   │   ├── experts/            # goal / macro / equity / portfolio / compliance 节点
│   │   └── supervisor/         # graph.py（LangGraph）· planner.py
│   ├── compliance/             # suitability.py（C-R 硬门）· language.py（误导用语检测）
│   ├── guardrails/             # input.py · output.py · process.py
│   ├── portfolio/              # metrics.py（纯函数）· optimize.py（风险预算）
│   ├── providers/              # SampleProvider · AkShareProvider（Protocol）· consensus · registry
│   ├── rag/                    # embed（本地哈希）· store（内存余弦）· corpus
│   └── security/               # sanitize.py（注入检测）· redact.py（PII 脱敏）
└── tests/                      # pytest（370 passed，conftest 强制隔离开发者 .env）
```

## 路线图 / 诚实的留白

已交付：

- ✅ Supervisor + 5 专家 LangGraph 流水线（目标 / 宏观 / 权益 / 风险组合 / 合规）
- ✅ 多源共识（支柱一）+ 多模型陪审（支柱二）
- ✅ 三层护栏（输入 / 过程 / 输出）+ 预算护栏
- ✅ 中国投资者适当性（C1–C5）+ 跨境汇率规则 + 误导用语检测
- ✅ A/港/美股覆盖（样例 Provider + AkShare 真实 Provider 骨架）
- ✅ RAG 宏观上下文 + 合规/研报语料（内存、本地哈希嵌入）
- ✅ Langfuse 全链路可观测（可选、离线安全）
- ✅ SSE 工作台 + 运行审计存储（memory / sqlite）
- ✅ 多套件评测门禁（64 例，含端到端 status_routing 与 allocation_sanity 硬门）
- ✅ Docker + docker-compose + CI（GitHub Actions）
- ✅ 持久化（RunStore：memory + sqlite，Postgres 预留）

已知留白与后续工作：

- **AkShare 列名校准** —— `src/wealthwise/providers/akshare_provider.py` 中每处列名假设都标了 `TODO(live-calibration)`，投产前需对着真实 AkShare 输出核对。经 keyed 验证：基金（`fund_open_fund_daily_em`）与宏观（`macro_china_lpr`）实时可达且列名已核，A 股实时行情 `stock_zh_a_spot_em` 的主机在验证网络里被 SSL 挡住（环境限制），详见 [`docs/real-data-verification.md`](docs/real-data-verification.md)。
- **辩论 / 层级式多智能体** —— 当前陪审是扁平多数投票；结构化辩论回环（Supervisor 调解专家分歧）是后续工作。
- **Postgres 运行存储** —— schema 已在 `store.py` 预留，当前仅实现 `memory` 与 `sqlite`。
- **真实协方差数据** —— 组合优化器用的是样例 Provider 的玩具级风险估计。真实均值方差 / Black-Litterman 需要来自真实价格历史的协方差矩阵。
- **付费数据源** —— 面向中国市场的付费量化数据（如 Wind、iFinD）未接入；AkShare Provider 仅覆盖公开数据。

## 许可

[MIT](LICENSE) (c) 2026 Kevin Tu（WealthWise / 武道AI）
