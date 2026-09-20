# InferenceLab

[![CI](https://github.com/asher0913/inference-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/asher0913/inference-lab/actions/workflows/ci.yml)

A reproducible model-serving experiment harness with async dynamic batching, semantic caching and
an OpenAI-compatible backend adapter.

## Architecture

```text
request -> semantic TTL/LRU cache -> bounded queue -> dynamic batch -> model backend -> metrics
```

- Batch size and queue-wait bounds expose the latency/utilization trade-off.
- Semantic cache supports TTL, LRU eviction and configurable similarity thresholds.
- Backend abstraction supports deterministic CI tests and real vLLM/SGLang/Ollama endpoints.
- Load generator reports throughput, P50/P95, cache hit rate, backend batches and mean batch size.

## Quick start

```bash
uv sync --extra dev
uv run pytest
uv run inference-bench --requests 80 --concurrency 16
uv run uvicorn inference_lab.api:app --reload
```

The default backend is deterministic and downloads no model. It validates scheduler and cache
behavior but its throughput is **not** an LLM performance claim.

## Connect a real server

```python
from inference_lab.backend import OpenAICompatibleBackend
from inference_lab.service import InferenceService

backend = OpenAICompatibleBackend(
    base_url="http://localhost:8000",
    model="Qwen/Qwen2.5-1.5B-Instruct",
)
service = InferenceService(backend, max_batch_size=8, max_wait_ms=6)
```

For credible resume metrics, record model, accelerator, prompt/output length distribution, warmup,
cache threshold and at least three repeated runs.

## Production roadmap

- TTFT and inter-token latency for streaming generation.
- Prefix/KV cache and GPU memory/SM utilization metrics.
- Admission control, cancellation and per-tenant quotas.
- Prometheus/OpenTelemetry export and Grafana dashboards.

## License

MIT

