"""Text sanitization + prompt-injection defense (deterministic, offline).

Two jobs:
1. User-input facing: `normalize` + `detect_injection` catch attacks in
   investor goals, queries, and any other user-controlled strings (block
   outright before they reach the advisory pipeline).
2. LLM-data facing: `neutralize_untrusted` de-fangs *external* text (market
   commentary, news headlines, analyst reports) before embedding it in agent
   prompts, so a malicious data source cannot hijack advisory verdicts.

All rule-based on purpose: eval and tests stay reproducible with no model
calls.
"""
import re
import unicodedata

# zero-width + BOM chars used to smuggle instructions past naive regexes
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

# Unambiguous attacks — used for BOTH seed screening and data neutralization.
_ATTACK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b"
        r"(previous|prior|above|earlier|all|these|any|the)\b.{0,25}\b"
        r"(instruction|direction|directive|rule|prompt|context|guardrail|policy)s?\b",
        re.I), "instruction_override"),
    (re.compile(
        r"\b(act as|you are now|pretend to be|role[- ]?play as|behave as)\b.{0,25}\b"
        r"(system|admin|administrator|developer|dan|root|assistant)\b",
        re.I), "role_hijack"),
    (re.compile(
        r"\bsystem prompt\b|\bdeveloper mode\b|\bjailbreak\b|\bDAN mode\b|"
        r"\bdo anything now\b",
        re.I), "jailbreak"),
    (re.compile(r"</?s>|<\|.*?\|>|\{\{.*?\}\}|\[/?INST\]|```", re.I),
     "delimiter_injection"),
    (re.compile(
        r"忽略[^。]{0,12}(之前|上述|前面|所有|以上)[^。]{0,12}"
        r"(指令|规则|提示|要求|设定)"), "instruction_override_cn"),
    (re.compile(
        r"(扮演|充当|假装|现在是).{0,8}(系统|管理员|开发者|助手)"), "role_hijack_cn"),
]

# Imperative verdict-forcing — only for data neutralization
# (would false-positive on legitimate user input if used for seed screening).
_VERDICT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"\b(respond|reply|output|answer|return|mark|classify|set|say|print)\b\s+"
        r"(with\s+|as\s+)?[\"']?\b(allow|approved|compliant|safe|buy|enter|low)\b",
        re.I), "verdict_injection"),
]


def normalize(text: str) -> str:
    """NFKC-normalize, strip zero-width/BOM chars, collapse whitespace."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    return re.sub(r"\s+", " ", t).strip()


def detect_injection(text: str) -> tuple[bool, str]:
    """Return (is_injection, category) for user-controlled input.

    Normalizes first so zero-width / homoglyph evasion is defeated.
    Returns (False, "") for benign input.
    """
    norm = normalize(text)
    for pat, cat in _ATTACK_PATTERNS:
        if pat.search(norm):
            return True, cat
    return False, ""


def neutralize_untrusted(text: str, label: str = "UNTRUSTED") -> str:
    """Wrap external text as untrusted data and strip embedded instructions.

    Redacts attack + verdict-forcing phrases, escapes tag characters so the
    payload cannot close the wrapper, and fences the result in
    <label>…</label> so the agent's system prompt can treat it strictly as
    passive data.
    """
    norm = normalize(text)
    for pat, _ in (*_ATTACK_PATTERNS, *_VERDICT_PATTERNS):
        norm = pat.sub("[removed]", norm)
    safe = norm.replace("<", "‹").replace(">", "›")
    return f"<{label}>{safe}</{label}>"
