"""Bootstrap — offline and runtime dependency factories.

Mirror of shopscout.bootstrap.

OfflineJuryClient: deterministic stage-aware verdicts without any API key.
build_sample_deps: fully offline stack for tests, demos, evals.
build_runtime_deps: real providers when use_real_providers=True + keys present;
    falls back to sample stack otherwise.
"""
from __future__ import annotations

from wealthwise.config import Settings, get_settings
from wealthwise.agents.deps import AdvisoryDeps
from wealthwise.llm import Verdict
from wealthwise.providers.sample import SampleFXProvider, SampleMacroProvider, SampleMarketProvider
from wealthwise.rag.corpus import load_policy_retriever, load_research_retriever
from wealthwise.rag.embed import LocalHashingEmbedder


def _evidence(user: str, labels: list[str]) -> str:
    """Strip instruction lines that merely enumerate the label vocabulary.

    A prompt line like "Classify: PASS, DOWNGRADE, or REJECT?" is a question,
    not evidence; keyword heuristics must not read it as a verdict.
    """
    keep: list[str] = []
    for line in user.splitlines():
        low = line.lower()
        named = sum(1 for l in labels if l.lower() in low)
        if named >= 2 and ("classify" in low or "respond" in low or low.endswith("?")):
            continue
        keep.append(line)
    return "\n".join(keep)


class OfflineJuryClient:
    """Deterministic offline judge — stage/context-aware, no API calls.

    Mirrors shopscout.bootstrap.OfflineJuryClient with wealthwise label sets.
    """

    def __init__(self, name: str = "offline") -> None:
        self.name = name

    def judge(self, system: str, user: str, labels: list[str]) -> Verdict:
        """Return a deterministic verdict from the allowed label set.

        Heuristics (in priority order):
        - Compliance labels ["PASS", "DOWNGRADE", "REJECT"]:
            → REJECT if user content signals cross-border violation, C1/C2 + R5,
              etc.  NOTE: we match only against the *user* portion so that
              instruction words in the system prompt (e.g. "REJECTED") do NOT
              accidentally trigger REJECT on every call.
            → DOWNGRADE if user content signals mismatch / de-risk needed
            → PASS otherwise
        - Macro/equity tilt labels ["overweight", "neutral", "underweight"]:
            → neutral (safe default)
        - Fallback: first label in the set.
        """
        # Match triggers against the *evidence* only. Excluding the system prompt
        # was not enough: the user prompt ends with its own instruction line
        # ("Classify: PASS, DOWNGRADE, or REJECT?"), which contains every label
        # name, so a naive substring scan returned DOWNGRADE for every single
        # call. That stayed invisible while portfolios were cash-heavy and the
        # jury almost never ran; it downgraded every advisory once they were not.
        user_lower = _evidence(user, labels).lower()

        if set(labels) == {"PASS", "DOWNGRADE", "REJECT"}:
            # Compliance judgment — check user content only.
            # "Suitability check: REJECT/DOWNGRADE" in the user prompt is the most
            # reliable signal.  We also check specific violation keywords.
            suitability_line = ""
            for line in user.splitlines():
                if "suitability check:" in line.lower():
                    suitability_line = line.lower()
                    break

            if (
                "suitability check: reject" in suitability_line
                or any(t in user_lower for t in ("cross-border violation",
                                                  "unauthorized: ", "prohibited"))
            ):
                label = "REJECT"
            elif (
                "suitability check: downgrade" in suitability_line
                or any(t in user_lower for t in ("downgrade", "mismatch",
                                                  "unsuitable", "too risky",
                                                  "above ceiling", "shortfall:"))
            ):
                label = "DOWNGRADE"
            else:
                label = "PASS"

        elif labels == ["overweight", "neutral", "underweight"]:
            # Macro tilt judgment — same evidence-only scoping as compliance
            combined = _evidence(user, labels).lower()
            if "underweight" in combined or "bear" in combined or "recession" in combined:
                label = "underweight"
            elif "overweight" in combined or "bull" in combined:
                label = "overweight"
            else:
                label = "neutral"

        else:
            # Generic: return first label
            label = labels[0]

        return Verdict(label=label, rationale=f"offline rule selected {label!r}")


def build_sample_deps(settings: Settings | None = None) -> AdvisoryDeps:
    """Build a fully offline AdvisoryDeps for tests, demos, and evals.

    Uses SampleMarket/Macro/FXProvider, OfflineJuryClient (×2), and the
    local hashing embedder — no network, no API keys.
    """
    s = settings or get_settings()
    data_dir = s.sample_data_dir
    embedder = LocalHashingEmbedder(dim=s.embed_dim)

    return AdvisoryDeps(
        market=SampleMarketProvider(data_dir),
        macro=SampleMacroProvider(data_dir),
        fx=SampleFXProvider(data_dir),
        jury_clients=[
            OfflineJuryClient("offline-a"),
            OfflineJuryClient("offline-b"),
        ],
        policy_retriever=load_policy_retriever(data_dir, embedder),
        research_retriever=load_research_retriever(data_dir, embedder),
        embedder=embedder,
        max_fx_exposure=s.max_fx_exposure,
        risk_budget_method=s.risk_budget_method,
        max_llm_judgments=s.max_llm_judgments,
    )


def build_runtime_deps(settings: Settings | None = None) -> AdvisoryDeps:
    """Build AdvisoryDeps using real providers when configured; else sample stack.

    Behaviour mirrors shopscout.bootstrap.build_runtime_deps:
    - use_real_providers=False (or missing keys) → delegates to build_sample_deps.
    - use_real_providers=True + both GLM and SiliconFlow keys set
        → AkShare market provider + real jury clients + SiliconFlow embedder.
    """
    s = settings or get_settings()

    if not s.use_real_providers:
        return build_sample_deps(s)

    # Real providers path — only reachable in production with keys
    # Guard: fall back if keys are absent
    has_jury_keys = bool(s.glm_api_key and s.siliconflow_api_key)
    if not has_jury_keys:
        return build_sample_deps(s)

    from wealthwise.crosscheck.jury import build_jury_clients
    from wealthwise.providers.tencent_provider import TencentMarketProvider
    from wealthwise.providers.universe import Universe

    # Real embedder (SiliconFlow) — imported lazily so offline tests never touch it
    try:
        from wealthwise.rag.backends import build_embedder
        embedder = build_embedder(s)
    except Exception:
        embedder = LocalHashingEmbedder(dim=s.embed_dim)

    data_dir = s.sample_data_dir
    base_embedder_offline = LocalHashingEmbedder(dim=s.embed_dim)

    return AdvisoryDeps(
        # Quote-based provider over qt.gtimg.cn. Replaces the AkShare eastmoney
        # screener, which resolves and handshakes but is reset mid-stream on
        # roughly two calls in three — see providers/tencent_provider.py.
        market=TencentMarketProvider(Universe.load()),
        macro=SampleMacroProvider(data_dir),  # AkShare macro not yet wired
        fx=SampleFXProvider(data_dir),         # AkShare FX not yet wired
        jury_clients=build_jury_clients(s),
        policy_retriever=load_policy_retriever(data_dir, base_embedder_offline),
        research_retriever=load_research_retriever(data_dir, base_embedder_offline),
        embedder=embedder,
        max_fx_exposure=s.max_fx_exposure,
        risk_budget_method=s.risk_budget_method,
        max_llm_judgments=s.max_llm_judgments,
    )
