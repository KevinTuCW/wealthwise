import pytest

from wealthwise.llm import Verdict, FakeModelClient, ModelClient, parse_verdict

LABELS = ["LOW", "MEDIUM", "HIGH"]


def test_fake_client_satisfies_protocol_and_returns_verdict():
    v = Verdict(label="LOW", rationale="low risk profile matches product")
    c = FakeModelClient(name="fake", verdict=v)
    assert isinstance(c, ModelClient)
    assert c.judge("sys", "user", LABELS) is v


def test_parse_plain_json():
    v = parse_verdict('{"label": "HIGH", "rationale": "aggressive growth portfolio"}', LABELS)
    assert v.label == "HIGH"
    assert v.rationale == "aggressive growth portfolio"


def test_parse_strips_code_fences():
    raw = '```json\n{"label": "MEDIUM", "rationale": "balanced risk"}\n```'
    assert parse_verdict(raw, LABELS).label == "MEDIUM"


def test_parse_label_is_case_insensitive_but_normalized():
    assert parse_verdict('{"label": "high", "rationale": "x"}', LABELS).label == "HIGH"


def test_parse_rejects_unknown_label():
    with pytest.raises(ValueError):
        parse_verdict('{"label": "EXTREME", "rationale": "x"}', LABELS)


def test_parse_rejects_non_json():
    with pytest.raises(ValueError):
        parse_verdict("I think the risk is high.", LABELS)
