"""Frozen numeric GSM8K scoring; no reference-answer-aware extraction."""
from fractions import Fraction
import re

NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?"


def normalize(value):
    try:
        return Fraction(value.replace(",", "").replace(" ", "").strip().rstrip("."))
    except (ValueError, ZeroDivisionError):
        return None


def extract_answer(text, strict=True):
    patterns = [r"####\s*\$?\s*(" + NUMBER + r")",
                r"\\boxed\{\s*\$?\s*(" + NUMBER + r")\s*\}",
                r"(?:final\s+answer|answer)\s*(?:is\s*)?[:=]\s*\**\s*\$?\s*(" + NUMBER + r")"]
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, flags=re.I)]
    if matches:
        return normalize(min(matches, key=lambda match: match.start()).group(1))
    if not strict:
        values = re.findall(NUMBER, text)
        if values:
            return normalize(values[-1])
    return None


def score(text, gold):
    expected = normalize(gold)
    if expected is None:
        raise ValueError("invalid gold answer")
    strict, flexible = extract_answer(text), extract_answer(text, strict=False)
    return dict(strict_correct=strict == expected, flexible_correct=flexible == expected,
                strict_answer=None if strict is None else str(strict),
                flexible_answer=None if flexible is None else str(flexible), gold=str(expected))
