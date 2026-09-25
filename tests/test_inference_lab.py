import asyncio

import httpx
import pytest

from inference_lab.backend import DeterministicBackend, OpenAICompatibleBackend
from inference_lab.benchmark import run_benchmark
from inference_lab.cache import SemanticCache
from inference_lab.cache_eval import LEXICAL, dataset, evaluate
from inference_lab.embedding import HashingEmbedder, literal_guard
from inference_lab.models import GenerationRequest
from inference_lab.service import InferenceService
from inference_lab.simulate import CostModel, run_static, simulate, workload


async def test_dynamic_batching_combines_concurrent_requests():
    backend = DeterministicBackend(base_latency_ms=1, per_item_ms=0)
    service = InferenceService(backend, max_batch_size=8, max_wait_ms=10)
    responses = await asyncio.gather(*(service.generate(GenerationRequest(f"unique prompt {i}")) for i in range(12)))
    await service.close()
    assert max(r.batch_size for r in responses) > 1
    assert len(backend.batch_sizes) < len(responses)


async def test_semantic_cache_skips_backend_on_repeat():
    backend = DeterministicBackend(base_latency_ms=1, per_item_ms=0)
    service = InferenceService(backend)
    first = await service.generate(GenerationRequest("explain retrieval"))
    second = await service.generate(GenerationRequest("explain retrieval"))
    await service.close()
    assert not first.cached and second.cached
    assert len(backend.batch_sizes) == 1


@pytest.mark.parametrize(("coalesce", "generations"), [(False, 16), (True, 1)])
async def test_identical_burst_is_coalesced(coalesce, generations):
    backend = DeterministicBackend(base_latency_ms=5, per_item_ms=0)
    service = InferenceService(backend, max_batch_size=32, max_wait_ms=2, coalesce=coalesce)
    responses = await asyncio.gather(*(service.generate(GenerationRequest("same prompt")) for _ in range(16)))
    await service.close()
    assert sum(backend.batch_sizes) == generations
    assert {r.text for r in responses} == {"answer:same prompt"}
    assert sum(r.coalesced for r in responses) == 16 - generations


async def test_coalesced_waiters_see_the_leaders_error():
    class Failing(DeterministicBackend):
        async def generate_batch(self, prompts, max_tokens, temperatures=None):
            await asyncio.sleep(0.005)
            raise RuntimeError("backend down")

    service = InferenceService(Failing(), max_wait_ms=1)
    calls = [service.generate(GenerationRequest("p")) for _ in range(4)]
    results = await asyncio.gather(*calls, return_exceptions=True)
    await service.close()
    assert all(isinstance(r, RuntimeError) for r in results)
    assert not service._inflight


async def test_answers_are_not_shared_across_output_limits_or_tenants():
    backend = DeterministicBackend(base_latency_ms=1, per_item_ms=0)
    service = InferenceService(backend)
    short = await service.generate(GenerationRequest("explain retrieval", max_tokens=4))
    long = await service.generate(GenerationRequest("explain retrieval", max_tokens=64))
    other_tenant = await service.generate(GenerationRequest("explain retrieval", max_tokens=64, tenant="b"))
    repeat = await service.generate(GenerationRequest("explain retrieval", max_tokens=64))
    await service.close()
    assert (short.text, long.text) == ("answer:expl", "answer:explain retrieval")
    assert not long.cached and not other_tenant.cached and repeat.cached
    assert sum(backend.batch_sizes) == 3


async def test_identical_prompts_with_different_limits_are_not_coalesced():
    backend = DeterministicBackend(base_latency_ms=5, per_item_ms=0)
    service = InferenceService(backend, max_batch_size=8, max_wait_ms=5)
    responses = await asyncio.gather(
        service.generate(GenerationRequest("same prompt", max_tokens=4)),
        service.generate(GenerationRequest("same prompt", max_tokens=64)),
    )
    await service.close()
    assert [r.text for r in responses] == ["answer:same", "answer:same prompt"]
    assert not any(r.coalesced for r in responses)


async def test_sampled_requests_bypass_cache_and_coalescing():
    backend = DeterministicBackend(base_latency_ms=5, per_item_ms=0)
    service = InferenceService(backend, max_batch_size=8, max_wait_ms=2)
    responses = await asyncio.gather(
        *(service.generate(GenerationRequest("tell a story", temperature=0.8)) for _ in range(4))
    )
    again = await service.generate(GenerationRequest("tell a story", temperature=0.8))
    await service.close()
    assert sum(backend.batch_sizes) == 5
    assert not any(r.cached or r.coalesced for r in [*responses, again])


def test_cache_ttl_and_lru():
    now = [0.0]
    cache = SemanticCache(capacity=2, ttl_seconds=10, threshold=0.99, clock=lambda: now[0])
    cache.put("alpha beta", "A")
    cache.put("gamma delta", "G")
    assert cache.get("alpha beta") == "A"  # now most recent
    cache.put("epsilon zeta", "E")  # evicts gamma delta
    assert cache.get("gamma delta") is None and len(cache) == 2
    now[0] = 11
    assert cache.get("alpha beta") is None and len(cache) == 0


def test_word_order_is_invisible_to_unigrams_but_not_to_bigrams():
    a, b = "convert 5 km to miles", "convert 5 miles to km"
    uni, bi = HashingEmbedder(ngram=1), HashingEmbedder(ngram=2)
    assert sum(x * y for x, y in zip(uni.encode(a), uni.encode(b), strict=True)) == pytest.approx(1.0)
    assert sum(x * y for x, y in zip(bi.encode(a), bi.encode(b), strict=True)) < 0.9


def test_literal_guard():
    assert literal_guard("what's france's capital", "what is the capital of france")  # apostrophes are not quotes
    assert literal_guard("translate to french: 'good morning'", "translate 'good morning' into french")
    assert not literal_guard("translate 'thank you' into french", "translate 'good morning' into french")
    assert not literal_guard("convert 12 km to miles", "convert 5 km to miles")
    assert not literal_guard("list foods that do not contain gluten", "list foods that contain gluten")


def test_cache_evaluation_headline():
    cached, queries = dataset()
    assert len(cached) == 37 and len(queries) == 267
    result = evaluate(LEXICAL)["configs"]
    default = result["unigrams (default)"]
    assert default["at_default_threshold_0.96"]["hits"] == 0
    assert default["at_default_threshold_0.96"]["wrong"] == 5  # swapped unit conversions
    assert default["best_with_no_wrong_answers"] is None
    assert result["uni+bigrams"]["at_default_threshold_0.96"]["wrong"] == 0


def test_simulation_ordering_and_padding_waste():
    requests = workload(8, 600, seed=0)
    seq, static, cont = (simulate(p, requests) for p in ("sequential", "static", "continuous"))
    assert seq["throughput_rps"] < static["throughput_rps"] < cont["throughput_rps"] <= 8.5
    assert cont["latency_p95_s"] < static["latency_p95_s"] < seq["latency_p95_s"]
    assert static["slot_utilisation"] < 0.5 and cont["slot_utilisation"] == 1.0
    assert seq == run_static(requests, CostModel(), max_batch=1, max_wait_ms=0.0)


def test_light_load_is_served_at_offered_rate():
    requests = workload(0.5, 400, seed=1)
    for policy in ("static", "continuous"):
        assert simulate(policy, requests)["throughput_rps"] == pytest.approx(0.5, rel=0.15)


async def test_openai_backend_sends_one_request_per_prompt():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        seen.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": f"echo {len(seen)}"}}]})

    backend = OpenAICompatibleBackend("http://vllm:8000/", "m", transport=httpx.MockTransport(handler))
    out = await backend.generate_batch(["a", "b", "c"], 16)
    assert len(out) == 3 and len(seen) == 3 and backend.batch_sizes == [3]
    assert all('"max_tokens":16' in body.replace(" ", "") for body in seen)
    seen.clear()
    await backend.generate_batch(["a", "b"], [5, 50], temperatures=[0.0, 0.7])
    bodies = sorted(body.replace(" ", "") for body in seen)
    assert any('"max_tokens":5,' in b and '"temperature":0.0' in b for b in bodies)
    assert any('"max_tokens":50,' in b and '"temperature":0.7' in b for b in bodies)


async def test_loadtest_counts():
    without = await run_benchmark(80, 16, coalesce=False)
    with_ = await run_benchmark(80, 16, coalesce=True)
    assert (without.backend_generations, without.coalesced_requests) == (16, 0)
    assert (with_.backend_generations, with_.coalesced_requests) == (3, 13)
    assert without.backend_batches >= 2 and with_.backend_batches >= 1  # batch counts depend on timing
    assert without.cache_hit_rate == with_.cache_hit_rate == 0.8


def test_api_generate_then_cached():
    from fastapi.testclient import TestClient

    from inference_lab.api import create_app

    with TestClient(create_app()) as client:
        first = client.post("/v1/generate", json={"prompt": "hello there"}).json()
        second = client.post("/v1/generate", json={"prompt": "hello there"}).json()
    assert not first["cached"] and second["cached"]
