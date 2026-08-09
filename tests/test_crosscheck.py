import pytest

from wealthwise.crosscheck import deliberate, JuryResult
from wealthwise.llm import Verdict, FakeModelClient

LABELS = ["LOW", "MEDIUM", "HIGH"]


def client(name, label):
    return FakeModelClient(name=name, verdict=Verdict(label=label, rationale=name))


def test_empty_raises():
    with pytest.raises(ValueError):
        deliberate([], "sys", "user", LABELS)


def test_single_client_capped_confidence_and_escalates():
    res = deliberate([client("glm", "LOW")], "sys", "user", LABELS)
    assert res.label == "LOW"
    assert res.disagreement is False
    assert res.confidence == 0.5
    assert res.escalate is True                  # 0.5 < 0.66


def test_unanimous_two_high_confidence_no_escalate():
    res = deliberate([client("glm", "LOW"), client("deepseek", "LOW")],
                     "sys", "user", LABELS)
    assert res.label == "LOW"
    assert res.disagreement is False
    assert res.confidence == 1.0
    assert res.escalate is False


def test_two_way_split_ties_to_none_and_escalates():
    res = deliberate([client("glm", "LOW"), client("deepseek", "HIGH")],
                     "sys", "user", LABELS)
    assert res.label is None                      # no strict majority
    assert res.disagreement is True
    assert res.confidence == 0.5
    assert res.escalate is True


def test_majority_of_three_holds_without_escalation():
    res = deliberate([client("a", "HIGH"), client("b", "HIGH"), client("c", "LOW")],
                     "sys", "user", LABELS)
    assert res.label == "HIGH"
    assert res.disagreement is True               # not unanimous
    assert res.confidence == pytest.approx(2 / 3)
    assert res.escalate is False                  # 0.667 >= 0.66
    assert res.sources == ["a", "b", "c"]
    assert len(res.verdicts) == 3


def test_tokens_are_summed():
    c1 = FakeModelClient("m1", Verdict(label="HIGH", rationale="r1", tokens=100))
    c2 = FakeModelClient("m2", Verdict(label="HIGH", rationale="r2", tokens=250))
    res = deliberate([c1, c2], "sys", "user", LABELS)
    assert res.tokens == 350
