"""Thin wrapper around the Anthropic SDK.

The API key is read from the ANTHROPIC_API_KEY environment variable.
"""
from __future__ import annotations

import logging
from typing import Iterator, Protocol

log = logging.getLogger("rag.llm")


class LLMError(Exception):
    """Raised for any provider failure, with a message safe to show to callers."""


class LLM(Protocol):
    configured: bool

    def complete(self, *, model: str, system: str, user: str, max_tokens: int) -> str: ...

    def stream(self, *, model: str, system: str, messages: list[dict], max_tokens: int) -> Iterator[str]: ...


class AnthropicLLM:
    def __init__(self, api_key: str):
        self.configured = bool(api_key)
        self._client = None
        if self.configured:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key, max_retries=3, timeout=60.0)

    def _require(self):
        if self._client is None:
            raise LLMError("ANTHROPIC_API_KEY is not set on the server.")
        return self._client

    def complete(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        import anthropic

        client = self._require()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIError as exc:
            raise LLMError(f"Model provider error: {exc.__class__.__name__}") from exc
        return "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()

    def stream(self, *, model: str, system: str, messages: list[dict], max_tokens: int) -> Iterator[str]:
        import anthropic

        client = self._require()
        try:
            with client.messages.stream(
                model=model, max_tokens=max_tokens, system=system, messages=messages
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except anthropic.APIError as exc:
            raise LLMError(f"Model provider error: {exc.__class__.__name__}") from exc
