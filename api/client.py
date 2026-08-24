from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError


class ChatAPIError(RuntimeError):
    """A user-facing error from the chat provider."""


class OpenAIClient:
    def __init__(self, api_key: str | None, base_url: str | None = None) -> None:
        if not api_key:
            raise ChatAPIError("OPENAI_API_KEY is not configured")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    @staticmethod
    def count_tokens(messages: Iterable[dict[str, str]]) -> int:
        """Return a dependency-free estimate suitable for context display."""
        return sum(max(1, len(message.get("content", "")) // 4) for message in messages)

    async def stream_chat(
        self,
        messages: Iterable[dict[str, str]],
        model: str,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content if chunk.choices else None
                if content:
                    yield content
        except RateLimitError as exc:
            raise ChatAPIError("Rate limit reached. Please wait and try again.") from exc
        except APIConnectionError as exc:
            raise ChatAPIError("Network connection failed. Check your connection and retry.") from exc
        except APIStatusError as exc:
            detail = getattr(exc, "message", None) or "The OpenAI request failed."
            raise ChatAPIError(detail) from exc
        except Exception as exc:
            raise ChatAPIError(f"Unexpected API error: {exc}") from exc

    async def close(self) -> None:
        await self._client.close()
