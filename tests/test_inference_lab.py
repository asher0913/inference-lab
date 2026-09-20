import asyncio

import pytest

from inference_lab.backend import DeterministicBackend
from inference_lab.models import GenerationRequest
from inference_lab.service import InferenceService


@pytest.mark.asyncio
async def test_dynamic_batching_combines_concurrent_requests() -> None:
    backend = DeterministicBackend(base_latency_ms=1, per_item_ms=0)
    service = InferenceService(backend, max_batch_size=8, max_wait_ms=10)
    responses = await asyncio.gather(
        *(service.generate(GenerationRequest(f"unique prompt {index}")) for index in range(12))
    )
    await service.close()
    assert max(response.batch_size for response in responses) > 1
    assert len(backend.batch_sizes) < len(responses)


@pytest.mark.asyncio
async def test_semantic_cache_skips_backend_on_repeat() -> None:
    backend = DeterministicBackend(base_latency_ms=1, per_item_ms=0)
    service = InferenceService(backend)
    first = await service.generate(GenerationRequest("explain retrieval"))
    second = await service.generate(GenerationRequest("explain retrieval"))
    await service.close()
    assert not first.cached
    assert second.cached
    assert len(backend.batch_sizes) == 1

