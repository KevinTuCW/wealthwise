"""Tests for Settings config — Task 1.1."""


def test_settings_defaults():
    from wealthwise.config import Settings

    s = Settings()
    assert s.llm_provider == "glm"
    assert s.llm_model == "glm-4.7"
    assert s.glm_base_url == "https://api.z.ai/api/paas/v4/"
    assert s.siliconflow_base_url == "https://api.siliconflow.com/v1"
    assert s.crosscheck_model == "deepseek-ai/DeepSeek-V3"
    assert s.use_real_providers is False
    assert s.sample_data_dir == "data/samples"
    assert s.enable_langfuse_tracing is False
    assert s.max_fx_exposure == 0.5
    assert s.max_llm_judgments == 12
    assert s.risk_budget_method == "risk_parity"
    assert s.token_price_per_1k == 0.0002
    assert s.run_store == "memory"


def test_tracing_disabled_by_default():
    from wealthwise.config import Settings

    s = Settings()
    assert s.tracing_enabled is False


def test_tracing_enabled_when_all_langfuse_fields_set():
    from wealthwise.config import Settings

    s = Settings(
        enable_langfuse_tracing=True,
        langfuse_public_key="pk-test-xxx",
        langfuse_secret_key="sk-test-xxx",
    )
    assert s.tracing_enabled is True


def test_tracing_disabled_when_flag_true_but_keys_missing():
    from wealthwise.config import Settings

    # flag on but keys absent — should still be False
    s = Settings(enable_langfuse_tracing=True, langfuse_public_key="", langfuse_secret_key="")
    assert s.tracing_enabled is False


def test_tracing_disabled_when_flag_false_but_keys_present():
    from wealthwise.config import Settings

    # keys provided but flag off — should still be False
    s = Settings(
        enable_langfuse_tracing=False,
        langfuse_public_key="pk-test-xxx",
        langfuse_secret_key="sk-test-xxx",
    )
    assert s.tracing_enabled is False


def test_get_settings_returns_singleton():
    from wealthwise.config import get_settings

    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
