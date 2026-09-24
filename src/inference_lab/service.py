from __future__ import annotations

import asyncio

from .backend import ModelBackend
from .batcher import DynamicBatcher
from .cache import SemanticCache
from .models import GenerationRequest, GenerationResponse


def _key(prompt: str) -> str:
    return " ".join(prompt.lower().split())


class InferenceService:
    """Semantic cache, then request coalescing, then the dynamic batcher.

    Coalescing (single-flight): while a prompt is being generated, identical
    prompts wait for that result instead of going to the backend again. Without
    it, a burst of identical requests all miss the still-empty cache.
    """

    def __init__(
        self,
        backend: ModelBackend,
        max_batch_size: int = 8,
        max_wait_ms: float = 8,
        cache: SemanticCache | None = None,
        coalesce: bool = True,
    ) -> None:
        self.batcher = DynamicBatcher(backend, max_batch_size, max_wait_ms)
        self.cache = cache or SemanticCache()
        self.coalesce = coalesce
        self._inflight: dict[str, asyncio.Future[GenerationResponse]] = {}
        self.stats = {"cache_hits": 0, "coalesced": 0, "generated": 0}

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        cached = self.cache.get(request.prompt)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return GenerationResponse(text=cached, cached=True, queue_ms=0.0, inference_ms=0.0, batch_size=0)
        key = _key(request.prompt)
        if self.coalesce and key in self._inflight:
            self.stats["coalesced"] += 1
            leader = await asyncio.shield(self._inflight[key])
            return GenerationResponse(leader.text, False, leader.queue_ms, leader.inference_ms, leader.batch_size, True)
        future: asyncio.Future[GenerationResponse] = asyncio.get_running_loop().create_future()
        if self.coalesce:
            self._inflight[key] = future
        try:
            self.stats["generated"] += 1
            response = await self.batcher.submit(request.prompt, request.max_tokens)
            self.cache.put(request.prompt, response.text)
            future.set_result(response)
            return response
        except BaseException as error:
            future.set_exception(error)
            future.exception()  # mark retrieved when nobody else was waiting
            raise
        finally:
            self._inflight.pop(key, None)

    async def close(self) -> None:
        await self.batcher.close()
