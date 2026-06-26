"""Shared LLM client helpers: plain text calls and strict-JSON classifier calls.

Provider-agnostic by design (app.config.LLM_PROVIDER): defaults to Claude (Anthropic), but
falls back to Groq's free tier when only a GROQ_API_KEY is configured, so the pipeline can be
exercised without Anthropic credits.
"""
import json
import re

from app import config

_clients: dict[str, object] = {}


def _build_client(model: str, temperature: float):
    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model, temperature=temperature, api_key=config.GROQ_API_KEY, timeout=120, max_retries=3
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model, temperature=temperature, api_key=config.ANTHROPIC_API_KEY, timeout=120, max_retries=3
    )


def get_client(model: str, temperature: float = 0.0):
    key = f"{config.LLM_PROVIDER}:{model}:{temperature}"
    if key not in _clients:
        _clients[key] = _build_client(model, temperature)
    return _clients[key]


def call_text(model: str, prompt: str, temperature: float = 0.0) -> str:
    client = get_client(model, temperature)
    response = client.invoke(prompt)
    return response.content if isinstance(response.content, str) else str(response.content)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def call_json(model: str, prompt: str) -> dict:
    """Call Claude expecting a single JSON object back; tolerate minor formatting noise."""
    raw = call_text(model, prompt, temperature=0.0)
    match = _JSON_RE.search(raw)
    if not match:
        raise ValueError(f"No JSON object found in model response: {raw!r}")
    return json.loads(match.group(0))
