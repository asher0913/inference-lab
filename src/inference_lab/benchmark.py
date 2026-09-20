from __future__ import annotations

import argparse
import asyncio
import json
import time
from statistics import mean

from .backend import DeterministicBackend
from .models import BenchmarkSummary, GenerationRequest
from .service import InferenceService


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


async def run_benchmark(requests: int, concurrency: int) -> BenchmarkSummary:
    backend = DeterministicBackend()
    service = InferenceService(backend, max_batch_size=8, max_wait_ms=6)
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    cache_hits = 0
    prompts = [
        "summarize checkout incident",
        "explain vector retrieval",
        "write a rollback checklist",
        "summarize checkout incident",
    ]

    async def one(index: int) -> None:
        nonlocal cache_hits
        async with semaphore:
            started = time.perf_counter()
            response = await service.generate(GenerationRequest(prompts[index % len(prompts)]))
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dynamic batching and semantic cache")
    parser.add_argument("--requests", type=int, default=80)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output", default="benchmark-report.json")
    args = parser.parse_args()
    summary = asyncio.run(run_benchmark(args.requests, args.concurrency))
    payload = summary.__dict__
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

