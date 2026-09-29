"""Shared, reference-free stopping after an explicitly completed numeric answer."""
import re

NUM = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?"
END = r"(?:\r?\n|[ \t]+(?=[^/\s])|[.!](?=[^\d])|\*\*|[}$])"
MARKED = re.compile(r"(?:####|(?:final\s+answer|^answer)\s*(?:is\s*)?[:=])\s*\**\s*\$?\s*" + NUM + END, re.I | re.M)
BOXED = re.compile(r"\\boxed\{\s*\$?\s*" + NUM + r"\s*\}", re.I)


def answer_complete(text):
    """Wait for a delimiter: a partial integer/decimal is not a finished answer."""
    return bool(MARKED.search(text) or BOXED.search(text))


def token_answer_complete(tokenizer, tokens):
    return answer_complete(tokenizer.decode(tokens, skip_special_tokens=True))
