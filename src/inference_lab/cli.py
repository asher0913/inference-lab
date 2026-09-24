"""inference-lab report | cache-eval"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .benchmark import run_benchmark
from .cache_eval import LEXICAL, evaluate, neural_configs
from .simulate import sweep

# Batch counts depend on how fast the event loop fills the 6 ms window, so they are left out.
DETERMINISTIC_LOADTEST_FIELDS = (
    "requests",
    "concurrency",
    "cache_hit_rate",
    "coalesced_requests",
    "backend_generations",
)


def loadtest() -> dict:
    out = {}
    for name, coalesce in (("without coalescing", False), ("with coalescing", True)):
        summary = asyncio.run(run_benchmark(80, 16, coalesce)).__dict__
        out[name] = {k: summary[k] for k in DETERMINISTIC_LOADTEST_FIELDS}
    return out


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="inference-lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="batching simulation, lexical cache evaluation, load-test counts")
    rep.add_argument("--out", type=Path, default=Path("results"))
    cev = sub.add_parser("cache-eval", help="cache evaluation including a neural embedder (needs '.[neural]')")
    cev.add_argument("--out", type=Path, default=Path("results/cache_minilm.json"))
    args = parser.parse_args(argv)
    if args.command == "report":
        _write(args.out / "batching.json", sweep())
        _write(args.out / "cache_lexical.json", evaluate(LEXICAL))
        _write(args.out / "loadtest.json", loadtest())
    else:
        _write(args.out, evaluate({**LEXICAL, **neural_configs()}))


if __name__ == "__main__":
    main()
