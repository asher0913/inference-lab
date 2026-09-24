from __future__ import annotations

import asyncio
from typing import Protocol


class ModelBackend(Protocol):
    async def generate_batch(self, prompts: list[str], max_tokens: int) -> list[str]: ...


class DeterministicBackend:
    """Small backend for CI and local load tests without downloading a model."""

    def __init__(self, base_latency_ms: float = 12, per_item_ms: float = 3) -> None:
        self.base_latency_ms = base_latency_ms
        self.per_item_ms = per_item_ms
        self.batch_sizes: list[int] = []

    async def generate_batch(self, prompts: list[str], max_tokens: int) -> list[str]:
        self.batch_sizes.append(len(prompts))
        await asyncio.sleep((self.base_latency_ms + self.per_item_ms * len(prompts)) / 1000)
        return [f"answer:{prompt[:max_tokens]}" for prompt in prompts]


class OpenAICompatibleBackend:
    """Adapter for vLLM, SGLang, Ollama and other OpenAI-style ``/v1/chat/completions`` servers.

    A batch is sent as concurrent requests; the server's own scheduler (continuous
    batching in vLLM and SGLang) decides how they share the GPU.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "local", transport=None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.transport = transport  # an httpx transport; tests pass httpx.MockTransport
        self.batch_sizes: list[int] = []

    async def generate_batch(self, prompts: list[str], max_tokens: int) -> list[str]:
        import httpx

        self.batch_sizes.append(len(prompts))
        async with httpx.AsyncClient(timeout=120, transport=self.transport) as client:
            responses = await asyncio.gather(
                *[
                    client.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                        },
                    )
                    for prompt in prompts
                ]
            )
        for response in responses:
            response.raise_for_status()
        return [response.json()["choices"][0]["message"]["content"] for response in responses]
