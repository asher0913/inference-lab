from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_tokens: int = 64


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    cached: bool
    queue_ms: float
    inference_ms: float
    batch_size: int


@dataclass(frozen=True)
class BenchmarkSummary:
    requests: int
    concurrency: int
    throughput_rps: float
    p50_ms: float
    p95_ms: float
    cache_hit_rate: float
    backend_batches: int
    mean_batch_size: float

