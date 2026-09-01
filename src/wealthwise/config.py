"""Wealthwise application settings — all env-configurable values live here.

Domain rule constants that are NOT env-driven (e.g. misleading-term blacklist,
restricted market lists) belong in their own modules — not here.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (used from Phase 2 onward; declared now so .env stays stable) ---
    llm_provider: str = "glm"                             # OpenAI-compatible endpoint
    llm_model: str = "glm-4.7"
    glm_api_key: str = ""
    glm_base_url: str = "https://api.z.ai/api/paas/v4/"  # z.ai OpenAI-compat gateway
    # cross-check second model (SiliconFlow international .com)
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.com/v1"
    crosscheck_model: str = "deepseek-ai/DeepSeek-V3"    # cross-lab judge, distinct from GLM
    # Third juror, also on SiliconFlow. Three labs (Zhipu / DeepSeek / Ant) give
    # the vote an actual majority: two models can only agree (confidence 1.0) or
    # tie (label=None), so "多数标签胜出" described something that never happened.
    # An odd jury makes 3/3, 2/3 and no-majority three distinct outcomes. Set to
    # "" to fall back to the two-model configuration.
    #
    # Was moonshotai/Kimi-K3, which dominated advisory latency at 7–47s per call
    # with no upper bound worth trusting. Ling-flash-2.0 returned identical
    # verdicts on all eight benchmark cases — four clean portfolios it correctly
    # passed, four violating ones it correctly caught — at ~1.6s. Kimi-K2.6 is
    # equally fast and was rejected: it downgraded two of the four clean
    # portfolios. Since the jury can only tighten a verdict, a juror biased
    # toward DOWNGRADE never leaks, it just routes every clean advisory to human
    # review, which fails the product instead of the investor.
    third_model: str = "inclusionAI/Ling-flash-2.0"
    # Jurors classify into a closed label set, where extended reasoning cost 30–60s
    # per call without changing the verdict. Off by default; set false to restore
    # it if a future juror is found to need it.
    llm_disable_thinking: bool = True
    llm_timeout: float = 60.0          # per-juror; deliberate() waits on all of them

    # --- embeddings ---
    embed_provider: str = "local"            # "local" (offline hashing) | "siliconflow"
    embed_model: str = "Qwen/Qwen3-Embedding-8B"
    embed_dim: int = 256

    # --- data providers ---
    use_real_providers: bool = False          # offline sample stack unless keys + this are set
    sample_data_dir: str = "data/samples"

    # --- equity ranking ---
    # Multi-factor scoring (value / momentum / low-vol / size / liquidity, see
    # portfolio/factors.py). Off by default, and the default is the honest one:
    # the weights are a house view rather than a backtested result, so the
    # legible "biggest first, cheapest breaks ties" rule stays the shipping
    # default until someone has validated the model on this universe.
    enable_factor_scoring: bool = False
    # Names whose price two feeds disagree on beyond the consensus tolerance are
    # dropped from selection rather than ranked. Set false to keep them (they
    # stay tagged either way, so the disagreement is never invisible).
    drop_on_data_disagreement: bool = True

    # --- observability (Langfuse — leave empty keys to run offline / no-op) ---
    enable_langfuse_tracing: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://us.cloud.langfuse.com"

    # --- domain thresholds / guardrail params ---
    max_fx_exposure: float = 0.5             # max fraction of portfolio in non-CNY assets
    max_llm_judgments: int = 12              # hard budget cap on LLM calls per advisory run
    risk_budget_method: str = "risk_parity" # "risk_parity" | "equal_weight" | "mean_variance"

    # --- cost accounting ---
    token_price_per_1k: float = 0.0002      # blended $/1k tokens (set to contract price)

    # --- persistence ---
    run_store: str = "memory"       # "memory" | "sqlite" | "postgres" — run/audit store
    run_store_path: str = "data/runs.db"    # SQLite file path (used when run_store="sqlite")
    # libpq connection string, used when run_store="postgres". Deliberately has
    # no default: one pointing at localhost would let a misconfigured deployment
    # start up happily and write its audit log where nobody is reading.
    run_store_dsn: str = ""

    @property
    def tracing_enabled(self) -> bool:
        """True only when tracing is explicitly enabled AND both Langfuse keys are set."""
        return (
            self.enable_langfuse_tracing
            and bool(self.langfuse_public_key)
            and bool(self.langfuse_secret_key)
        )

    # Vendor-specific switch that turns off extended reasoning. A juror picks one
    # label from a closed set; measured on this workload, GLM-4.7 with thinking on
    # took 30–60s and ~1,150 completion tokens to reach the same verdict it reaches
    # in 2.4s and 42 tokens with it off. The two gateways spell the field
    # differently, hence one per client rather than a shared constant.
    def _no_thinking(self, vendor: str) -> dict | None:
        if not self.llm_disable_thinking:
            return None
        return ({"thinking": {"type": "disabled"}} if vendor == "zai"
                else {"enable_thinking": False})

    def primary_client_kwargs(self) -> dict:
        """Kwargs to build the primary (GLM) model client."""
        return {
            "name": self.llm_model,
            "model": self.llm_model,
            "base_url": self.glm_base_url,
            "api_key": self.glm_api_key,
            "extra_body": self._no_thinking("zai"),
            "timeout": self.llm_timeout,
        }

    def crosscheck_client_kwargs(self) -> dict:
        """Kwargs to build the cross-check (SiliconFlow / DeepSeek) model client."""
        return {
            "name": self.crosscheck_model,
            "model": self.crosscheck_model,
            "base_url": self.siliconflow_base_url,
            "api_key": self.siliconflow_api_key,
            "extra_body": self._no_thinking("siliconflow"),
            "timeout": self.llm_timeout,
        }

    def third_client_kwargs(self) -> dict:
        """Kwargs to build the third juror (SiliconFlow), making the vote odd."""
        return {
            "name": self.third_model,
            "model": self.third_model,
            "base_url": self.siliconflow_base_url,
            "api_key": self.siliconflow_api_key,
            "extra_body": self._no_thinking("siliconflow"),
            "timeout": self.llm_timeout,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
