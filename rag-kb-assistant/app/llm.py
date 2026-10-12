```python
"""Thin wrapper around the OpenAI SDK.

The API key is read from the OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import logging
from typing import Iterator, Protocol

log = logging.getLogger("rag.llm")


class LLMError(Exception):
    """Raised for provider failures with a safe message for callers."""


class LLM(Protocol):
    configured: bool

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
    ) -> str:
        ...

    def stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int,
    ) -> Iterator[str]:
        ...


class OpenAILLM:
    def __init__(self, api_key: str):
        self.configured = bool(api_key)
        self._client = None

        if self.configured:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key,
                max_retries=3,
                timeout=60.0,
            )

    def _require(self):
        if self._client is None:
            raise LLMError(
                "OPENAI_API_KEY is not set on the server."
            )

        return self._client

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
    ) -> str:
        from openai import APIError

        client = self._require()

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": user,
                    },
                ],
            )

            content = response.choices[0].message.content

            return (content or "").strip()

        except APIError as exc:
            log.exception("OpenAI completion failed")

            raise LLMError(
                f"Model provider error: {exc.__class__.__name__}"
            ) from exc

    def stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int,
    ) -> Iterator[str]:
        from openai import APIError

        client = self._require()

        try:
            stream = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": system,
                    },
                    *messages,
                ],
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    yield content

        except APIError as exc:
            log.exception("OpenAI streaming failed")

            raise LLMError(
                f"Model provider error: {exc.__class__.__name__}"
            ) from exc
```
