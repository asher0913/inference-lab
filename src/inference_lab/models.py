from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0
    tenant: str = "default"


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    cached: bool
    queue_ms: float
    inference_ms: float
    batch_size: int
    coalesced: bool = False


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
    coalesced_requests: int = 0
    backend_generations: int = 0
