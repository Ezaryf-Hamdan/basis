"""PII redaction for anything that leaves the process.

Lifted from ``otel_setup.py`` (the ``_EMAIL_RE`` / ``_PHONE_RE`` /
``redact_pii`` block), which is one of the better-thought-through pieces in the
source repo: it recognises that prompts and completions get written to a trace
backend and that raw customer text must not go there.

Kept from the original:
  * the email and phone patterns, unchanged,
  * ``redact_pii(None) -> ""`` (call sites rely on it),
  * the 8000-char truncation cap with an explicit ``...[TRUNCATED]`` marker,
    rather than silent slicing.

Added, because a delivery platform handling client documents will hit these:
  * AWS access key IDs and long bearer/secret-looking tokens,
  * IBANs and long digit runs that read as payment card numbers,
  * a ``redact`` entry point that honours the ``redact_pii`` setting so a
    consumer can turn the whole thing off deliberately rather than by omission.

This module has no dependencies beyond the standard library, so it is usable
from a consumer that installs basis without the ``otel`` extra.
"""
from __future__ import annotations

import re

from ..settings import settings

__all__ = ["MAX_CONTENT_CHARS", "redact", "redact_pii", "truncate"]

MAX_CONTENT_CHARS = 8000

_EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")
_PHONE_RE = re.compile(r"\+?\d{1,4}[-.\s]\(?\d{1,4}\)?[-.\s]\d{1,4}[-.\s]\d{1,9}")

# Credentials. An agent that reads client config documents will encounter these,
# and a leaked key in a span is worse than a leaked email.
_AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_BEARER_RE = re.compile(r"(?i)\b(bearer|token|secret|password|api[_-]?key)\b\s*[:=]\s*\S+")
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)

# Financial identifiers - SAP delivery documents are full of them.
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")

# Order matters: private keys before bearer tokens (a key block contains no
# delimiter but is longer), and both before phone, so a long digit run inside a
# credential is not partially rewritten first.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (_PRIVATE_KEY_RE, "<PRIVATE_KEY>"),
    (_AWS_KEY_RE, "<AWS_KEY>"),
    (_BEARER_RE, r"\1: <REDACTED>"),
    (_EMAIL_RE, "<EMAIL>"),
    (_IBAN_RE, "<IBAN>"),
    (_CARD_RE, "<CARD>"),
    (_PHONE_RE, "<PHONE>"),
)


def redact_pii(text: str | None) -> str:
    """Replace identifiers and credentials with placeholders.

    Returns "" for None, matching the lifted contract.
    """
    if not text:
        return ""
    out = str(text)
    for pattern, replacement in _RULES:
        out = pattern.sub(replacement, out)
    return out


def redact(text: str | None) -> str:
    """``redact_pii``, unless redaction is switched off in settings.

    Prefer this at call sites. The explicit setting means "we send raw text to
    the trace backend" is a decision someone made, not a line nobody wrote.
    """
    if not settings().redact_pii:
        return "" if not text else str(text)
    return redact_pii(text)


def truncate(text: str, max_chars: int | None = None) -> str:
    """Cap length with a visible marker, as the original did."""
    limit = max_chars if max_chars is not None else settings().max_span_content_chars
    if len(text) <= limit:
        return text
    return text[:limit] + "...[TRUNCATED]"
