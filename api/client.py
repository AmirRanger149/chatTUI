from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable

import httpx


class ChatAPIError(RuntimeError):
    """A user-facing error from the chat provider."""


class ChatAPIClient:
    """Native REST client for Google's Gemini generateContent API."""

    def __init__(self, api_key: str | None, base_url: str | None = None) -> None:
        if not api_key:
            raise ChatAPIError("GEMINI_API_KEY is not configured")
        self._api_key = api_key
        self._base_url = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self._client = httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(60.0, connect=15.0),
        )

    async def stream_chat(
        self,
        messages: Iterable[dict[str, str]],
        model: str,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        system_prompt = ""
        contents: list[dict[str, object]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_prompt = content
            else:
                contents.append({"role": "model" if role == "assistant" else "user", "parts": [{"text": content}]})
        payload: dict[str, object] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        try:
            url = f"{self._base_url}/models/{model}:streamGenerateContent"
            async with self._client.stream("POST", url, params={"alt": "sse", "key": self._api_key}, json=payload) as response:
                if response.status_code == 429:
                    raise ChatAPIError("Gemini rate limit reached. Please wait and try again.")
                if response.status_code >= 400:
                    raise ChatAPIError(await self._error_detail(response))
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        chunk = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    for candidate in chunk.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            if part.get("text"):
                                yield part["text"]
        except httpx.TimeoutException as exc:
            raise ChatAPIError("The Gemini request timed out. Check your connection and retry.") from exc
        except httpx.RequestError as exc:
            raise ChatAPIError("Network connection failed. Check your connection and retry.") from exc
        except ChatAPIError:
            raise
        except Exception as exc:
            raise ChatAPIError(f"Unexpected Gemini API error: {exc}") from exc

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def count_tokens(messages: Iterable[dict[str, str]]) -> int:
        return sum(max(1, len(message.get("content", "")) // 4) for message in messages)

    @staticmethod
    async def _error_detail(response: httpx.Response) -> str:
        try:
            body = json.loads(await response.aread())
            return body.get("error", {}).get("message", "The Gemini API request failed.")
        except (json.JSONDecodeError, TypeError, AttributeError):
            return f"The Gemini API request failed (HTTP {response.status_code})."
