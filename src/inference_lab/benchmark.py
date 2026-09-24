"""The load test behind the headline numbers: 80 requests, 16 at a time, through cache + batcher."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from statistics import mean

from .backend import DeterministicBackend, OpenAICompatibleBackend
from .models import BenchmarkSummary, GenerationRequest
from .service import InferenceService

PROMPTS = [
    "summarize checkout incident",
    "explain vector retrieval",
    "write a rollback checklist",
    "summarize checkout incident",
]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


async def run_benchmark(requests: int, concurrency: int, coalesce: bool = True, backend=None) -> BenchmarkSummary:
    backend = backend or DeterministicBackend()
    service = InferenceService(backend, max_batch_size=8, max_wait_ms=6, coalesce=coalesce)
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    cache_hits = 0

    async def one(index: int) -> None:
        nonlocal cache_hits
        async with semaphore:
            started = time.perf_counter()
            response = await service.generate(GenerationRequest(PROMPTS[index % len(PROMPTS)]))
            latencies.append((time.perf_counter() - started) * 1000)
            cache_hits += int(response.cached)

    started = time.perf_counter()
    await asyncio.gather(*(one(index) for index in range(requests)))
    elapsed = time.perf_counter() - started
    await service.close()
    batch_sizes = backend.batch_sizes
    return BenchmarkSummary(
        requests=requests,
        concurrency=concurrency,
        throughput_rps=requests / elapsed,
        p50_ms=_percentile(latencies, 0.50),
        p95_ms=_percentile(latencies, 0.95),
        cache_hit_rate=cache_hits / requests,
        backend_batches=len(batch_sizes),
        mean_batch_size=mean(batch_sizes) if batch_sizes else 0.0,
        coalesced_requests=service.stats["coalesced"],
        backend_generations=sum(batch_sizes),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load-test the cache, request coalescing and dynamic batcher")
    parser.add_argument("--requests", type=int, default=80)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--no-coalesce", action="store_true", help="disable single-flight request coalescing")
    parser.add_argument("--base-url", help="an OpenAI-compatible server (vLLM, SGLang, Ollama) instead of the stub")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", default="benchmark-report.json")
    args = parser.parse_args()
    backend = OpenAICompatibleBackend(args.base_url, args.model) if args.base_url else None
    summary = asyncio.run(run_benchmark(args.requests, args.concurrency, not args.no_coalesce, backend))
    payload = summary.__dict__
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
