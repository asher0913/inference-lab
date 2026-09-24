"""Render docs/batching.png and docs/cache.png from results/ (needs matplotlib)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"sequential": "#9e9e9e", "static": "#d95f02", "continuous": "#1b9e77"}


def batching() -> None:
    data = json.loads((ROOT / "results" / "batching.json").read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for policy, rows in data["curves"].items():
        rates = [r["rate_rps"] for r in rows]
        ax1.plot(rates, [r["latency_p95_s"] for r in rows], "o-", color=COLORS[policy], label=policy, ms=4, lw=2)
        ax2.plot(rates, [r["throughput_rps"] for r in rows], "o-", color=COLORS[policy], label=policy, ms=4, lw=2)
    ax1.axhline(data["slo_p95_latency_s"], color="black", ls=":", lw=1)
    ax1.text(0.6, data["slo_p95_latency_s"] * 1.15, "p95 SLO", fontsize=8)
    ax1.set_yscale("log")
    ax1.set_ylabel("p95 end-to-end latency (s)")
    ax2.plot([0, max(rates)], [0, max(rates)], color="black", ls=":", lw=1, label="offered load")
    ax2.set_ylabel("completed requests / s")
    for ax in (ax1, ax2):
        ax.set_xlabel("offered load (requests / s)")
        ax.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "batching.png", dpi=150)


def cache() -> None:
    data = json.loads((ROOT / "results" / "cache_minilm.json").read_text())
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    styles = {
        "unigrams (default)": ("#d95f02", "--"),
        "uni+bigrams": ("#7570b3", "-."),
        "MiniLM-L6": ("#66a61e", ":"),
        "MiniLM-L6 + literal guard": ("#1b9e77", "-"),
    }
    for name, (color, style) in styles.items():
        curve = data["configs"][name]["curve"]
        ax.plot([c["wrong"] for c in curve], [c["hit_rate_pct"] for c in curve], style, color=color, lw=2, label=name)
        default = data["configs"][name]["at_default_threshold_0.96"]
        ax.plot(default["wrong"], default["hit_rate_pct"], "o", color=color, ms=6)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel(f"wrong answers served (of {data['queries']} lookups)")
    ax.set_ylabel("paraphrases answered from cache (%)")
    ax.set_title("Threshold sweep 0.50 to 0.99; dots mark 0.96", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "cache.png", dpi=150)


if __name__ == "__main__":
    batching()
    cache()
    print("wrote docs/batching.png, docs/cache.png")
