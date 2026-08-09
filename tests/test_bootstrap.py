"""TDD tests for wealthwise.bootstrap — build_sample_deps + build_runtime_deps."""
from __future__ import annotations

import pytest


class TestBuildSampleDeps:
    def test_returns_advisory_deps(self):
        from wealthwise.agents.deps import AdvisoryDeps
        from wealthwise.bootstrap import build_sample_deps

        deps = build_sample_deps()
        assert isinstance(deps, AdvisoryDeps)

    def test_providers_are_sample_types(self):
        """Sample deps must use offline sample providers (not real AkShare)."""
        from wealthwise.providers.sample import (
            SampleFXProvider, SampleMacroProvider, SampleMarketProvider,
        )
        from wealthwise.bootstrap import build_sample_deps

        deps = build_sample_deps()
        assert isinstance(deps.market, SampleMarketProvider)
        assert isinstance(deps.macro, SampleMacroProvider)
        assert isinstance(deps.fx, SampleFXProvider)

    def test_jury_clients_are_offline(self):
        """Sample deps must use OfflineJuryClient (no real LLM keys)."""
        from wealthwise.bootstrap import OfflineJuryClient, build_sample_deps

        deps = build_sample_deps()
        assert len(deps.jury_clients) >= 1
        for client in deps.jury_clients:
            assert isinstance(client, OfflineJuryClient), (
                f"Expected OfflineJuryClient, got {type(client).__name__}"
            )

    def test_jury_clients_implement_model_client_protocol(self):
        from wealthwise.llm import ModelClient
        from wealthwise.bootstrap import build_sample_deps

        deps = build_sample_deps()
        for client in deps.jury_clients:
            assert isinstance(client, ModelClient), (
                f"{type(client).__name__} does not satisfy ModelClient protocol"
            )

    def test_retrievers_are_present(self):
        from wealthwise.bootstrap import build_sample_deps

        deps = build_sample_deps()
        assert deps.policy_retriever is not None
        assert deps.research_retriever is not None

    def test_retrievers_can_search(self):
        from wealthwise.bootstrap import build_sample_deps

        deps = build_sample_deps()
        policy_docs = deps.policy_retriever.search("适当性 风险", k=2)
        research_docs = deps.research_retriever.search("macro equity", k=2)
        assert isinstance(policy_docs, list)
        assert isinstance(research_docs, list)

    def test_embedder_is_present(self):
        from wealthwise.bootstrap import build_sample_deps

        deps = build_sample_deps()
        assert deps.embedder is not None

    def test_max_llm_judgments_positive(self):
        from wealthwise.bootstrap import build_sample_deps

        deps = build_sample_deps()
        assert deps.max_llm_judgments > 0

    def test_offline_jury_client_judge_returns_verdict(self):
        from wealthwise.bootstrap import OfflineJuryClient

        client = OfflineJuryClient("test")
        verdict = client.judge("system", "user text", ["PASS", "DOWNGRADE", "REJECT"])
        from wealthwise.llm import Verdict
        assert isinstance(verdict, Verdict)
        assert verdict.label in ("PASS", "DOWNGRADE", "REJECT")

    def test_offline_jury_client_respects_label_set(self):
        """OfflineJuryClient must only return labels from the provided label set."""
        from wealthwise.bootstrap import OfflineJuryClient

        client = OfflineJuryClient("test")
        labels_sets = [
            ["PASS", "DOWNGRADE", "REJECT"],
            ["overweight", "neutral", "underweight"],
            ["YES", "NO"],
        ]
        for labels in labels_sets:
            verdict = client.judge("sys", "user", labels)
            assert verdict.label in labels, (
                f"Returned label {verdict.label!r} not in {labels}"
            )


class TestBuildRuntimeDeps:
    def test_fallback_to_sample_when_use_real_providers_false(self):
        """build_runtime_deps(use_real_providers=False) must return sample deps."""
        from wealthwise.agents.deps import AdvisoryDeps
        from wealthwise.bootstrap import OfflineJuryClient, build_runtime_deps
        from wealthwise.config import Settings

        settings = Settings(use_real_providers=False)
        deps = build_runtime_deps(settings)

        assert isinstance(deps, AdvisoryDeps)
        # Should fall back to sample providers
        from wealthwise.providers.sample import SampleMarketProvider
        assert isinstance(deps.market, SampleMarketProvider)
        # Should use offline jury
        for client in deps.jury_clients:
            assert isinstance(client, OfflineJuryClient)

    def test_returns_advisory_deps_type(self):
        from wealthwise.agents.deps import AdvisoryDeps
        from wealthwise.bootstrap import build_runtime_deps
        from wealthwise.config import Settings

        deps = build_runtime_deps(Settings(use_real_providers=False))
        assert isinstance(deps, AdvisoryDeps)

    def test_no_settings_arg_uses_defaults(self):
        """build_sample_deps() with no args must not raise."""
        from wealthwise.bootstrap import build_sample_deps
        deps = build_sample_deps()
        assert deps is not None
