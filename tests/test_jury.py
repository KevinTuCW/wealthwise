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

    def fake_init(self, name, model, base_url, api_key, **kwargs):
        self.name = name
        self._model = model
        captured.append((name, model, base_url, api_key, kwargs))

    monkeypatch.setattr(OpenAICompatibleModelClient, "__init__", fake_init)
    return captured


def test_build_jury_clients_from_settings(monkeypatch):
    captured = _capture(monkeypatch)
    clients = jury.build_jury_clients(_settings(third_model="inclusionAI/Ling-flash-2.0"))

    assert [c.name for c in clients] == [
        "glm-4.7", "deepseek-ai/DeepSeek-V3", "inclusionAI/Ling-flash-2.0"]
    assert [c[:4] for c in captured] == [
        ("glm-4.7", "glm-4.7", "https://api.z.ai/api/paas/v4/", "k1"),
        ("deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3",
         "https://api.siliconflow.com/v1", "k2"),
        ("inclusionAI/Ling-flash-2.0", "inclusionAI/Ling-flash-2.0",
         "https://api.siliconflow.com/v1", "k2"),
    ]


def test_thinking_disabled_uses_each_vendors_own_field(monkeypatch):
    """z.ai and SiliconFlow spell the switch differently; sending the wrong one is a no-op."""
    captured = _capture(monkeypatch)
    jury.build_jury_clients(_settings(third_model="inclusionAI/Ling-flash-2.0"))

    assert captured[0][4]["extra_body"] == {"thinking": {"type": "disabled"}}   # z.ai
    for juror in captured[1:]:                                                  # SiliconFlow
        assert juror[4]["extra_body"] == {"enable_thinking": False}
    for juror in captured:
        assert juror[4]["timeout"] > 0


def test_thinking_can_be_re_enabled(monkeypatch):
    captured = _capture(monkeypatch)
    jury.build_jury_clients(_settings(llm_disable_thinking=False))
    assert all(juror[4]["extra_body"] is None for juror in captured)


def test_jury_is_odd_so_a_majority_can_exist(monkeypatch):
    _capture(monkeypatch)
    clients = jury.build_jury_clients(_settings(third_model="inclusionAI/Ling-flash-2.0"))
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


# ── jurors are polled concurrently ───────────────────────────────────────────

def test_deliberate_polls_jurors_concurrently():
    """A deliberation should cost the slowest juror, not the sum of all of them."""
    import time

    class SlowClient:
        def __init__(self, name, delay, label):
            self.name, self._delay, self._label = name, delay, label

        def judge(self, system, user, labels):
            time.sleep(self._delay)
            return Verdict(label=self._label, rationale="slow", tokens=1)

    clients = [SlowClient(f"j{i}", 0.3, "PASS") for i in range(3)]
    start = time.monotonic()
    result = deliberate(clients, "sys", "usr", LABELS)
    elapsed = time.monotonic() - start

    assert result.label == "PASS"
    # Sequential would be ~0.9s; concurrent ~0.3s. 0.6s separates them safely.
    assert elapsed < 0.6, f"jurors appear to run sequentially ({elapsed:.2f}s)"


def test_deliberate_keeps_client_order_regardless_of_finish_order():
    """Reconciliation reads verdicts positionally, so order must not follow latency."""
    import time

    class OrderedClient:
        def __init__(self, name, delay, label):
            self.name, self._delay, self._label = name, delay, label

        def judge(self, system, user, labels):
            time.sleep(self._delay)
            return Verdict(label=self._label, rationale=self.name, tokens=1)

    # Deliberately inverted: the first client finishes last.
    clients = [
        OrderedClient("slowest", 0.30, "REJECT"),
        OrderedClient("middle", 0.15, "PASS"),
        OrderedClient("fastest", 0.01, "PASS"),
    ]
    result = deliberate(clients, "sys", "usr", LABELS)

    assert result.sources == ["slowest", "middle", "fastest"]
    assert [v.rationale for v in result.verdicts] == ["slowest", "middle", "fastest"]
    assert result.label == "PASS"      # 2/3 majority, unaffected by finish order


def test_deliberate_propagates_a_juror_error():
    class Boom:
        name = "boom"

        def judge(self, system, user, labels):
            raise RuntimeError("juror exploded")

    clients = [FakeModelClient("ok", Verdict(label="PASS", rationale="")), Boom()]
    try:
        deliberate(clients, "sys", "usr", LABELS)
    except RuntimeError as exc:
        assert "juror exploded" in str(exc)
    else:
        raise AssertionError("a failing juror must not be silently dropped")
