"""OpenAI-compatible chat completion client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


class ChatClient(Protocol):
    """Protocol for chat-completion style clients."""

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        """Return the assistant text for a list of chat messages."""


@dataclass(slots=True)
class LLMClient:
    """Small OpenAI-compatible client configured by environment variables."""

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout: int = 120

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("LLM_API_KEY")
        self.base_url = (self.base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = self.model or os.getenv("LLM_MODEL") or "gpt-4.1-mini"
        if not self.api_key:
            raise ValueError("LLM_API_KEY is required to call the language model API.")

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        """Call an OpenAI-compatible `/chat/completions` endpoint."""
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError(
                "LLM API calls require the dependency: pip install requests"
            ) from exc

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        try:
            return str(payload["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM API response shape: {payload}") from exc
