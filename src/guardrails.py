import re
from dataclasses import dataclass


@dataclass
class GuardrailDecision:
    allowed: bool
    reason: str = ""
    message: str = ""


INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"ignore\s+all\s+instructions",
    r"disregard\s+the\s+system",
    r"reveal\s+system\s+prompt",
    r"print\s+the\s+hidden\s+prompt",
    r"jailbreak",
    r"developer\s+mode",
    r"bypass\s+safety",
    r"act\s+as\s+an\s+unfiltered",
]

SENSITIVE_PATTERNS = [
    r"admin\s+password",
    r"internal\s+credentials",
    r"database\s+dump",
    r"private\s+key",
    r"api\s+key",
    r"card\s+number",
    r"cvv",
]

UNSAFE_OUTPUT_PATTERNS = [
    r"<\|system\|>",
    r"ignore\s+all\s+instructions",
    r"admin\s+password",
    r"private\s+key",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in patterns)


def evaluate_input(query: str) -> GuardrailDecision:
    q = (query or "").strip()
    if not q:
        return GuardrailDecision(False, "empty_query", "Please enter a valid banking-related question.")

    if len(q) > 1200:
        return GuardrailDecision(
            False,
            "query_too_long",
            "Your request is too long. Please ask a shorter banking-related question.",
        )

    if _matches_any(q, INJECTION_PATTERNS):
        return GuardrailDecision(
            False,
            "prompt_injection_detected",
            "I cannot follow instruction-manipulation requests. Please ask a normal NUST Bank question.",
        )

    if _matches_any(q, SENSITIVE_PATTERNS):
        return GuardrailDecision(
            False,
            "sensitive_request_detected",
            "I cannot provide sensitive or confidential information.",
        )

    return GuardrailDecision(True)


def evaluate_output(answer: str) -> GuardrailDecision:
    a = (answer or "").strip()
    if not a:
        return GuardrailDecision(False, "empty_answer", "I could not generate a safe answer. Please rephrase your question.")

    if _matches_any(a, UNSAFE_OUTPUT_PATTERNS):
        return GuardrailDecision(
            False,
            "unsafe_output_detected",
            "I cannot provide that response. Please ask a standard NUST Bank question.",
        )

    return GuardrailDecision(True)
