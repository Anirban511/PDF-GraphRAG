"""
guardrails.py — Rule-based query safety filter (no LLM required).

Checks for two categories of bad input:
  1. Prompt injection — attempts to override system instructions
  2. Blocked content — known harmful request patterns

This runs in microseconds with zero external calls, which is preferable to
an LLM-based classifier that adds ~300 ms and requires a working model.

To disable entirely during development, set GUARDRAILS_ENABLED=false in .env.
"""

from __future__ import annotations
import re
from app.config import settings
from app.utils.logger import logger

# Patterns that try to hijack the system prompt or override instructions
_INJECTION = re.compile(
    r"ignore\s+(previous|all|prior|your)\s+instructions?"
    r"|you\s+are\s+now\s+"
    r"|disregard\s+(your|all|the)"
    r"|pretend\s+you\s+(are|have|can)"
    r"|jailbreak"
    r"|reveal\s+(your\s+)?(system\s+)?prompt"
    r"|forget\s+(everything|all|your\s+instructions)",
    re.IGNORECASE,
)

# Hard-blocked request categories
_BLOCKED_PHRASES = {
    "make a bomb",
    "synthesize explosives",
    "how to make drugs",
    "child pornography",
    "csam",
    "how to hack into",
    "create malware",
    "write ransomware",
    "ddos attack",
}


def is_query_safe(query: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    reason is an empty string when the query is safe.
    """
    q_lower = query.lower()

    if _INJECTION.search(query):
        reason = "Prompt injection pattern detected."
        logger.warning(f"Guardrail blocked (injection): {query[:80]}")
        return False, reason

    for phrase in _BLOCKED_PHRASES:
        if phrase in q_lower:
            reason = f"Request contains blocked content."
            logger.warning(f"Guardrail blocked (phrase '{phrase}'): {query[:80]}")
            return False, reason

    return True, ""
