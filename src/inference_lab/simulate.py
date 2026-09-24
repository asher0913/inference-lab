"""Discrete-event simulation of LLM request scheduling on one accelerator.

Three policies share one cost model:

``sequential``
    One request at a time.
``static``
    What :class:`~inference_lab.batcher.DynamicBatcher` does: collect up to
    ``max_batch`` requests (waiting at most ``max_wait_ms``), run the batch until
    its longest output is finished, and return every result together. Finished
    sequences keep their slot until then.
``continuous``
    Iteration-level scheduling as in Orca and vLLM: waiting requests join the
    running batch at the next decode step, and a sequence leaves as soon as its
    last token is produced.

The cost model is illustrative, not calibrated to a particular GPU. Decoding is
memory-bandwidth bound, so one decode step costs a fixed amount plus a little
per sequence, and prefill costs per prompt token. Chunked prefill and KV-cache
capacity limits are not modelled.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

POLICIES = ("sequential", "static", "continuous")


@dataclass(frozen=True)
class CostModel:
    step_ms: float = 6.0  # one decode step, whatever the batch size
    per_seq_ms: float = 0.25  # added per sequence in the batch
    prefill_ms_per_token: float = 0.05
    max_batch: int = 32

    def step(self, batch: int) -> float:
        return self.step_ms + self.per_seq_ms * batch


@dataclass(frozen=True)
class Request:
    id: int
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int


def workload(rate_rps: float, n: int, seed: int = 0) -> list[Request]:
    """Poisson arrivals; log-normal prompt (median 300) and output (median 120) lengths."""
    rng = random.Random(f"workload:{seed}:{rate_rps}")
    t, out = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate_rps) * 1000
        prompt = min(2048, max(16, round(rng.lognormvariate(math.log(300), 0.6))))
        output = min(1024, max(4, round(rng.lognormvariate(math.log(120), 0.9))))
        out.append(Request(i, t, prompt, output))
    return out


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def _summary(requests: list[Request], done: dict[int, float], busy_slots: float, used_slots: float) -> dict:
    latency = [done[r.id] - r.arrival_ms for r in requests]
    span_s = (max(done.values()) - requests[0].arrival_ms) / 1000
    return {
        "throughput_rps": round(len(requests) / span_s, 3),
        "output_tokens_per_s": round(sum(r.output_tokens for r in requests) / span_s, 1),
        "latency_p50_s": round(_percentile(latency, 0.5) / 1000, 3),
        "latency_p95_s": round(_percentile(latency, 0.95) / 1000, 3),
        "slot_utilisation": round(used_slots / busy_slots, 3) if busy_slots else 0.0,
    }


def run_static(requests: list[Request], cost: CostModel, max_batch: int, max_wait_ms: float) -> dict:
    waiting, t, done = deque(requests), 0.0, {}
    busy_slots = used_slots = 0.0
    while waiting:
        first = max(t, waiting[0].arrival_ms)
        deadline = first + max_wait_ms
        batch = []
        while waiting and len(batch) < max_batch and waiting[0].arrival_ms <= deadline:
            batch.append(waiting.popleft())
        start = max(first, batch[-1].arrival_ms) if len(batch) == max_batch else deadline
        start = first if max_wait_ms == 0 else start
        longest = max(r.output_tokens for r in batch)
        t = start + cost.prefill_ms_per_token * sum(r.prompt_tokens for r in batch) + longest * cost.step(len(batch))
        for r in batch:
            done[r.id] = t  # the batch returns together
        busy_slots += len(batch) * longest
        used_slots += sum(r.output_tokens for r in batch)
    return _summary(requests, done, busy_slots, used_slots)


def run_continuous(requests: list[Request], cost: CostModel) -> dict:
    waiting, running, t, done = deque(requests), [], 0.0, {}
    busy_slots = used_slots = 0.0
    while waiting or running:
        admitted = []
        while waiting and waiting[0].arrival_ms <= t and len(running) < cost.max_batch:
            r = waiting.popleft()
            running.append([r, r.output_tokens])
            admitted.append(r)
        if not running:
            t = waiting[0].arrival_ms
            continue
        if admitted:  # one step that also prefills the newcomers (and yields their first token)
            steps, dt = 1, cost.step(len(running)) + cost.prefill_ms_per_token * sum(r.prompt_tokens for r in admitted)
        else:  # jump ahead to the next finish or the next arrival that could join
            step_ms = cost.step(len(running))
            steps = min(left for _, left in running)
            if waiting and len(running) < cost.max_batch:
                steps = min(steps, max(1, math.ceil((waiting[0].arrival_ms - t) / step_ms)))
            dt = steps * step_ms
        t += dt
        busy_slots += steps * len(running)
        used_slots += steps * len(running)
        for item in running:
            item[1] -= steps
        for r, left in running:
            if left <= 0:
                done[r.id] = t
        running = [item for item in running if item[1] > 0]
    return _summary(requests, done, busy_slots, used_slots)


def simulate(policy: str, requests: list[Request], cost: CostModel | None = None, max_wait_ms: float = 6.0) -> dict:
    cost = cost or CostModel()
    if policy == "sequential":
        return run_static(requests, cost, max_batch=1, max_wait_ms=0.0)
    if policy == "static":
        return run_static(requests, cost, max_batch=cost.max_batch, max_wait_ms=max_wait_ms)
    if policy == "continuous":
        return run_continuous(requests, cost)
    raise ValueError(f"unknown policy {policy!r}")


def sweep(rates=(0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16), n: int = 1500, seeds=(0, 1, 2), slo_p95_s: float = 10.0) -> dict:
    cost = CostModel()
    curves = {}
    for policy in POLICIES:
        rows = []
        for rate in rates:
            runs = [simulate(policy, workload(rate, n, seed), cost) for seed in seeds]
            rows.append({"rate_rps": rate, **{k: round(sum(r[k] for r in runs) / len(runs), 3) for k in runs[0]}})
        curves[policy] = rows
    capacity = {
        p: max((row["rate_rps"] for row in rows if row["latency_p95_s"] <= slo_p95_s), default=0.0)
        for p, rows in curves.items()
    }
    return {
        "cost_model": cost.__dict__,
        "requests_per_point": n,
        "seeds": list(seeds),
        "slo_p95_latency_s": slo_p95_s,
        "max_rate_within_slo": capacity,
        "curves": curves,
    }
