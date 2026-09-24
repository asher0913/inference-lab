"""Does the semantic cache serve the right answer? A labelled set of paraphrases and near-misses.

Half of the base prompts of each family are cached. Every paraphrase of every
prompt is then looked up. A paraphrase of a cached prompt should hit and get
that prompt's answer; anything else that hits gets someone else's answer.
The near-misses are the dangerous part: swapped units, another language,
another country, or a negation. They change the answer while changing the
words very little.
"""

from __future__ import annotations

import itertools
import random

from .cache import SemanticCache
from .embedding import HashingEmbedder, literal_guard


def _family_units():
    pairs = [("km", "miles"), ("miles", "km"), ("kg", "pounds"), ("pounds", "kg"), ("liters", "gallons")]
    for n, (a, b) in itertools.product(["3", "5", "12", "40", "100"], pairs):
        yield (
            f"convert {n} {a} to {b}",
            [f"how many {b} is {n} {a}", f"{n} {a} in {b}", f"please convert {n} {a} into {b}"],
        )


def _family_translation():
    phrases = ["good morning", "thank you very much", "where is the train station", "I would like a coffee"]
    for p, lang in itertools.product(phrases, ["french", "german", "spanish", "japanese"]):
        yield (
            f"translate '{p}' into {lang}",
            [f"how do you say '{p}' in {lang}", f"translate to {lang}: '{p}'", f"'{p}' in {lang}, please"],
        )


def _family_code():
    tasks = [
        "sorts a list in ascending order",
        "sorts a list in descending order",
        "reverses a string",
        "checks whether a number is prime",
    ]
    for task, lang in itertools.product(tasks, ["python", "javascript", "go"]):
        yield (
            f"write a {lang} function that {task}",
            [f"{lang} function that {task}", f"implement a function in {lang} which {task}",
             f"can you write {lang} code that {task}"],
        )  # fmt: skip


def _family_facts():
    for attr, country in itertools.product(
        ["capital", "currency", "population"], ["france", "japan", "brazil", "canada", "egypt"]
    ):
        yield (
            f"what is the {attr} of {country}",
            [f"{attr} of {country}?", f"tell me the {attr} of {country}", f"what's {country}'s {attr}"],
        )


def _family_negation():
    for base in [
        "contain gluten",
        "do not contain gluten",
        "are high in protein",
        "are not high in protein",
        "are safe for dogs",
        "are not safe for dogs",
        "need refrigeration",
        "do not need refrigeration",
    ]:
        yield (
            f"list foods that {base}",
            [f"which foods {base}", f"give me a list of foods that {base}", f"name some foods that {base}"],
        )


FAMILIES = {
    "unit conversion": _family_units,
    "translation": _family_translation,
    "code": _family_code,
    "facts": _family_facts,
    "negation": _family_negation,
}

LEXICAL = {
    "unigrams (default)": (lambda: HashingEmbedder(ngram=1), False),
    "unigrams + literal guard": (lambda: HashingEmbedder(ngram=1), True),
    "uni+bigrams": (lambda: HashingEmbedder(ngram=2), False),
    "uni+bigrams + literal guard": (lambda: HashingEmbedder(ngram=2), True),
}


def neural_configs() -> dict:
    from .embedding import SentenceTransformerEmbedder

    model = SentenceTransformerEmbedder()
    return {"MiniLM-L6": (lambda: model, False), "MiniLM-L6 + literal guard": (lambda: model, True)}


def dataset(seed: int = 0):
    """(cached base prompts with answers, queries as (text, answer, family, answerable))."""
    rng = random.Random(seed)
    cached, queries = {}, []
    for family, make in FAMILIES.items():
        combos = list(make())
        if family == "negation":  # cache one polarity of each property, so the other is a near-miss
            keep = set(range(0, len(combos), 2))
        else:
            keep = set(rng.sample(range(len(combos)), len(combos) // 2))
        for i, (base, paraphrases) in enumerate(combos):
            if i in keep:
                cached[base] = base
            for text in paraphrases if i in keep else [base, *paraphrases]:
                queries.append((text, base, family, i in keep))
    return cached, queries


def _safe_point(curve: list[dict]) -> dict | None:
    safe = [c for c in curve if c["wrong"] == 0]
    return max(safe, key=lambda c: (c["hits"], -c["threshold"])) if safe else None


def evaluate(configs: dict | None = None, thresholds=None, seed: int = 0) -> dict:
    """Hit rate and wrong answers served across thresholds, overall and per family."""
    configs = configs or LEXICAL
    thresholds = thresholds or [round(0.5 + 0.01 * k, 2) for k in range(50)]
    cached, queries = dataset(seed)
    answerable = sum(q[3] for q in queries)
    per_family_answerable = {f: sum(q[3] for q in queries if q[2] == f) for f in FAMILIES}
    out = {"cached_prompts": len(cached), "queries": len(queries), "answerable_queries": answerable, "configs": {}}
    for name, (make_embedder, guarded) in configs.items():
        cache = SemanticCache(
            capacity=10_000, threshold=1.0, embedder=make_embedder(), guard=literal_guard if guarded else None
        )
        for prompt, answer in cached.items():
            cache.put(prompt, answer)
        scored = []
        for text, answer, family, _ in queries:
            entry, score = cache.lookup(text)
            scored.append((round(score, 6), entry is not None and entry.value == answer, family))

        def curve_for(rows, n_answerable):
            curve = []
            for t in thresholds:
                hits = [ok for score, ok, _ in rows if score >= t]
                curve.append(
                    {
                        "threshold": t,
                        "hits": sum(hits),
                        "hit_rate_pct": round(100 * sum(hits) / n_answerable, 1),
                        "wrong": len(hits) - sum(hits),
                    }
                )
            return curve

        curve = curve_for(scored, answerable)
        families = {}
        for f in FAMILIES:
            fam_curve = curve_for([s for s in scored if s[2] == f], per_family_answerable[f])
            families[f] = {
                "at_0.96": next(c for c in fam_curve if c["threshold"] == 0.96) if 0.96 in thresholds else None,
                "best_with_no_wrong_answers": _safe_point(fam_curve),
            }
        enabled = {f: v["best_with_no_wrong_answers"] for f, v in families.items() if v["best_with_no_wrong_answers"]}
        enabled = {f: point for f, point in enabled.items() if point["hits"] > 0}
        out["configs"][name] = {
            "per_family_thresholds": {
                "thresholds": {f: point["threshold"] for f, point in enabled.items()},
                "exact_match_only": [f for f in FAMILIES if f not in enabled],
                "hit_rate_pct": round(100 * sum(p["hits"] for p in enabled.values()) / answerable, 1),
                "wrong": 0,
            },
            "at_default_threshold_0.96": next((c for c in curve if c["threshold"] == 0.96), None),
            "best_with_no_wrong_answers": _safe_point(curve),
            "per_family": families,
            "curve": curve,
        }
    return out
