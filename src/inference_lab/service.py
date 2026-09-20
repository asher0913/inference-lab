from __future__ import annotations

from .backend import ModelBackend
from .batcher import DynamicBatcher
from .cache import SemanticCache
from .models import GenerationRequest, GenerationResponse


class InferenceService:
    def __init__(
        self,
        backend: ModelBackend,
        max_batch_size: int = 8,
        max_wait_ms: float = 8,
        cache: SemanticCache | None = None,
    ) -> None:
        self.batcher = DynamicBatcher(backend, max_batch_size, max_wait_ms)
        self.cache = cache or SemanticCache()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        cached = self.cache.get(request.prompt)
        if cached is not None:
            return GenerationResponse(
                text=cached,
                cached=True,
                queue_ms=0.0,
                inference_ms=0.0,
                batch_size=0,
            )
        response = await self.batcher.submit(request.prompt, request.max_tokens)
        self.cache.put(request.prompt, response.text)
        return response

    async def close(self) -> None:
        await self.batcher.close()

