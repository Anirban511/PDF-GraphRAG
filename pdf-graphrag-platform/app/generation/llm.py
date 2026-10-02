"""
llm.py — Multi-provider LLM interface.

WHY THIS EXISTS:
  The whole codebase calls call_llm() / stream_llm() and never imports a
  provider directly. That means the LLM backend is swappable from ONE place
  by changing a single setting — LLM_PROVIDER — with no other code changes.

PROVIDERS:
  "groq"      — hosted Groq API using OpenAI-compatible interface.
                Uses OpenAI GPT-OSS 120B by default.

  "ollama"    — local Ollama inference.

  "openai"    — hosted OpenAI API.

  "anthropic" — hosted Anthropic API.

DESIGN RATIONALE — local-first, API-optional:
  The project is built local-first for two design goals: zero running cost
  and full data privacy (financial documents are often confidential, and
  on-device inference means nothing leaves the machine). The tradeoff is the
  local model is weaker at structured extraction. Because the provider is
  abstracted here, the high-volume extraction stage can be routed to a cheap
  hosted model (e.g. gpt-4o-mini) for better accuracy and speed when those
  matter more than keeping data on-device — different stages, different needs.
"""

from __future__ import annotations
import json
import httpx
from app.config import settings
from app.utils.logger import logger


# ──────────────────────────────────────────────────────────────────────
# OLLAMA (local, default)
# ──────────────────────────────────────────────────────────────────────

def _ollama_payload(system: str, user: str, stream: bool) -> dict:
    return {
        "model": settings.llm_model,
        "stream": stream,
        "options": {"temperature": settings.temperature,
                    "num_predict": settings.max_tokens},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }


def _ollama_call(system: str, user: str) -> str:
    try:
        r = httpx.post(f"{settings.ollama_base_url}/api/chat",
                       json=_ollama_payload(system, user, False), timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot reach Ollama at {settings.ollama_base_url}. "
            "Start it with: ollama serve")


def _ollama_stream(system: str, user: str):
    try:
        with httpx.stream("POST", f"{settings.ollama_base_url}/api/chat",
                          json=_ollama_payload(system, user, True), timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if delta := chunk.get("message", {}).get("content"):
                    yield delta
                if chunk.get("done"):
                    break
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot reach Ollama at {settings.ollama_base_url}. "
            "Start it with: ollama serve")


# ──────────────────────────────────────────────────────────────────────
# OPENAI (hosted)
# ──────────────────────────────────────────────────────────────────────

def _openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=settings.openai_api_key)


def _openai_call(system: str, user: str) -> str:
    resp = _openai_client().chat.completions.create(
        model=settings.openai_model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


def _openai_stream(system: str, user: str):
    stream = _openai_client().chat.completions.create(
        model=settings.openai_model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
# ──────────────────────────────────────────────────────────────────────
# GROQ (hosted, OpenAI-compatible)
# ──────────────────────────────────────────────────────────────────────

def _groq_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        )

    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set."
        )

    return OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )


def _groq_call(system: str, user: str) -> str:
    resp = _groq_client().chat.completions.create(
        model=settings.groq_model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    return resp.choices[0].message.content


def _groq_stream(system: str, user: str):
    stream = _groq_client().chat.completions.create(
        model=settings.groq_model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content

        if delta:
            yield delta

# ──────────────────────────────────────────────────────────────────────
# ANTHROPIC (hosted)
# ──────────────────────────────────────────────────────────────────────

def _anthropic_client():
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _anthropic_call(system: str, user: str) -> str:
    resp = _anthropic_client().messages.create(
        model=settings.anthropic_model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def _anthropic_stream(system: str, user: str):
    with _anthropic_client().messages.stream(
        model=settings.anthropic_model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        yield from stream.text_stream


# ──────────────────────────────────────────────────────────────────────
# DISPATCH — the public interface the rest of the app uses
# ──────────────────────────────────────────────────────────────────────
_CALL = {
    "ollama": _ollama_call,
    "openai": _openai_call,
    "anthropic": _anthropic_call,
    "groq": _groq_call,
}

_STREAM = {
    "ollama": _ollama_stream,
    "openai": _openai_stream,
    "anthropic": _anthropic_stream,
    "groq": _groq_stream,
}

def call_llm(system: str, user: str, **_) -> str:
    """Blocking call routed to the configured provider."""
    provider = settings.llm_provider.lower()
    if provider not in _CALL:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. "
                         f"Use one of: {', '.join(_CALL)}")
    text = _CALL[provider](system, user)
    logger.debug(f"LLM[{provider}] response: {len(text)} chars")
    return text


def stream_llm(system: str, user: str):
    """Streaming call routed to the configured provider."""
    provider = settings.llm_provider.lower()
    if provider not in _STREAM:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. "
                         f"Use one of: {', '.join(_STREAM)}")
    yield from _STREAM[provider](system, user)
