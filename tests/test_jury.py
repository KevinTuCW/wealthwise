import wealthwise.crosscheck.jury as jury
from wealthwise.config import Settings
from wealthwise.crosscheck import deliberate
from wealthwise.llm import FakeModelClient, OpenAICompatibleModelClient, Verdict


def _settings(**kw):
    base = dict(glm_api_key="k1", llm_model="glm-4.7",
                glm_base_url="https://api.z.ai/api/paas/v4/",
                siliconflow_api_key="k2",
                crosscheck_model="deepseek-ai/DeepSeek-V3",
                siliconflow_base_url="https://api.siliconflow.com/v1")
    base.update(kw)
    return Settings(**base)


def _capture(monkeypatch):
    captured = []

    def fake_init(self, name, model, base_url, api_key):
        self.name = name
        self._model = model
        captured.append((name, model, base_url, api_key))

    monkeypatch.setattr(OpenAICompatibleModelClient, "__init__", fake_init)
    return captured


def test_build_jury_clients_from_settings(monkeypatch):
    captured = _capture(monkeypatch)
    clients = jury.build_jury_clients(_settings(third_model="moonshotai/Kimi-K3"))

    assert [c.name for c in clients] == [
        "glm-4.7", "deepseek-ai/DeepSeek-V3", "moonshotai/Kimi-K3"]
    assert captured == [
        ("glm-4.7", "glm-4.7", "https://api.z.ai/api/paas/v4/", "k1"),
        ("deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3",
         "https://api.siliconflow.com/v1", "k2"),
        ("moonshotai/Kimi-K3", "moonshotai/Kimi-K3",
         "https://api.siliconflow.com/v1", "k2"),
    ]


def test_jury_is_odd_so_a_majority_can_exist(monkeypatch):
    _capture(monkeypatch)
    clients = jury.build_jury_clients(_settings(third_model="moonshotai/Kimi-K3"))
    assert len(clients) % 2 == 1


def test_third_model_can_be_disabled(monkeypatch):
    _capture(monkeypatch)
    clients = jury.build_jury_clients(_settings(third_model=""))
    assert [c.name for c in clients] == ["glm-4.7", "deepseek-ai/DeepSeek-V3"]


# ── what an odd jury buys, in compliance terms ───────────────────────────────

LABELS = ["PASS", "DOWNGRADE", "REJECT"]


def _jurors(*labels):
    return [FakeModelClient(name=f"m{i}", verdict=Verdict(label=lb, rationale=lb))
            for i, lb in enumerate(labels)]


def test_two_of_three_is_a_real_majority():
    """The outcome a two-model jury could never produce."""
    r = deliberate(_jurors("DOWNGRADE", "DOWNGRADE", "PASS"), "sys", "user", LABELS)
    assert r.label == "DOWNGRADE"
    assert round(r.confidence, 3) == 0.667
    assert r.disagreement is True


def test_three_way_split_refuses_to_rule():
    r = deliberate(_jurors("PASS", "DOWNGRADE", "REJECT"), "sys", "user", LABELS)
    assert r.label is None
    assert r.escalate is True


def test_a_two_model_jury_has_no_middle_ground():
    """Why the third juror exists: two models only ever agree or tie."""
    agree = deliberate(_jurors("REJECT", "REJECT"), "sys", "user", LABELS)
    split = deliberate(_jurors("REJECT", "PASS"), "sys", "user", LABELS)
    assert (agree.label, agree.confidence) == ("REJECT", 1.0)
    assert split.label is None and split.confidence == 0.5


def test_budget_estimate_tracks_the_actual_jury_size():
    """The cost guard must scale with the jury, not assume two models.

    It was a hard-coded 2 ("2 clients × 1 call"), so a third juror would have
    left every budget projection understating the real spend by a third.
    """
    from dataclasses import replace

    from wealthwise.agents.supervisor.graph import _jury_calls
    from wealthwise.bootstrap import build_sample_deps

    deps = build_sample_deps()
    assert _jury_calls(replace(deps, jury_clients=_jurors("PASS", "PASS"))) == 2
    assert _jury_calls(replace(deps, jury_clients=_jurors("PASS", "PASS", "PASS"))) == 3
