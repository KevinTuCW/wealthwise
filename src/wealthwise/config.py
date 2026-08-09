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

    # --- embeddings ---
    embed_provider: str = "local"            # "local" (offline hashing) | "siliconflow"
    embed_model: str = "Qwen/Qwen3-Embedding-8B"
    embed_dim: int = 256

    # --- data providers ---
    use_real_providers: bool = False          # offline sample stack unless keys + this are set
    sample_data_dir: str = "data/samples"

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
    run_store: str = "memory"               # "memory" | "sqlite" — run/audit persistence

    @property
    def tracing_enabled(self) -> bool:
        """True only when tracing is explicitly enabled AND both Langfuse keys are set."""
        return (
            self.enable_langfuse_tracing
            and bool(self.langfuse_public_key)
            and bool(self.langfuse_secret_key)
        )

    def primary_client_kwargs(self) -> dict:
        """Kwargs to build the primary (GLM) model client."""
        return {
            "name": self.llm_model,
            "model": self.llm_model,
            "base_url": self.glm_base_url,
            "api_key": self.glm_api_key,
        }

    def crosscheck_client_kwargs(self) -> dict:
        """Kwargs to build the cross-check (SiliconFlow / DeepSeek) model client."""
        return {
            "name": self.crosscheck_model,
            "model": self.crosscheck_model,
            "base_url": self.siliconflow_base_url,
            "api_key": self.siliconflow_api_key,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
