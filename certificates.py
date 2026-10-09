#!/usr/bin/env python3
"""Shared cache, sampling and certificate routines from Experiment 2.
Bundled locally; no separate Experiment 2 installation is required.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

VERSION = "experiment2-v1"
MODELS = ["qwen2.5:3b-instruct", "qwen2.5:7b-instruct"]
EPSILONS = [0.1, 0.5, 1, 2, 5]
METHODS = ["theorem1", "theorem3"]
LABELS = {"theorem1": "Thm 1: simplex projection", "theorem3": "Thm 3: direct gap"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def slug(model):
    # Identical to Experiment 1's prediction-cache filename convention.
    return model.replace("/", "_").replace(":", "_") + "_" + digest(model)[:8]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def atomic_csv(rows, path):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False, encoding="utf-8")
    temp.replace(path)


def channel_params(channel, epsilon):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive; values above 1 are supported.")
    inverse = math.exp(-epsilon)  # Stable even for large positive epsilon.
    if channel == "GRR":
        p, q = 1 / (1 + 2 * inverse), inverse / (1 + 2 * inverse)
    elif channel == "OUE":
        p, q = 0.5, inverse / (1 + inverse)
    else:
        raise ValueError(f"Unknown channel: {channel}")
    if p <= q:
        raise ValueError("epsilon is too small for double-precision channel inversion; use a larger positive value.")
    return p, q


def draw_honest_counts(rng, n, f, p, q, channel, repetitions):
    """Exact aggregate sampling, preserving standard OUE coordinate dependence."""
    if channel == "GRR":
        return rng.multinomial(n, q + (p - q) * f, size=repetitions)
    category_counts = rng.multinomial(n, f, size=repetitions)
    return (rng.binomial(category_counts, p)
            + rng.binomial(n - category_counts, q))


def theorem1_bounds(s, N, p, q, bbar, delta):
    t = math.sqrt(math.log(6 / delta) / (2 * N))
    a = (1 - bbar) * (p - q)
    # Outward numerical expansion preserves conservativeness. Do not clip both
    # ends into [0,1], which would turn some empty regions into singletons.
    lo = np.maximum(0, (s - t - bbar - (1 - bbar) * q) / a - 1e-12)
    hi = np.minimum(1, (s + t - (1 - bbar) * q) / a + 1e-12)
    return lo, hi, t


def query_extrema(lo, hi):
    """Exact extrema of v1-v2 over coordinate bounds intersected with simplex."""
    lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
    feasible = ((lo <= hi).all(axis=1) & (lo.sum(axis=1) <= 1)
                & (hi.sum(axis=1) >= 1))

    def optimize(order):
        v = lo.copy()
        remaining = 1 - v.sum(axis=1)
        for i in order:
            step = np.minimum(np.maximum(remaining, 0), np.maximum(hi[:, i] - v[:, i], 0))
            v[:, i] += step
            remaining -= step
        return np.where(feasible, v[:, 0] - v[:, 1], np.nan)

    return optimize([1, 2, 0]), optimize([0, 2, 1]), feasible


def theorem3_bounds(s, N, p, q, bbar, delta, channel):
    L = math.log(2 / delta)
    nu = p + q if channel == "GRR" else 0.5
    rh = math.sqrt(2 * L / N)
    rb = 2 * L / (3 * N) + math.sqrt(2 * nu * L / N + 4 * L**2 / (9 * N**2))
    r = min(rh, rb)
    z = s[:, 0] - s[:, 1]
    a = (1 - bbar) * (p - q)
    lo = np.maximum(-1, (z - bbar - r) / a - 1e-12)
    hi = np.minimum(1, (z + bbar + r) / a + 1e-12)
    feasible = lo <= hi
    return (np.where(feasible, lo, np.nan), np.where(feasible, hi, np.nan),
            feasible, {"nu": nu, "r_hoeffding": rh, "r_bernstein": rb,
                       "r_direct": r, "radius_branch": "hoeffding" if rh <= rb else "bernstein"})


def decision(lo, hi, feasible):
    result = np.zeros(len(lo), dtype=np.int8)
    result[feasible & (lo > 1e-12)] = 1
    result[feasible & (hi < -1e-12)] = -1
    return result


def rate_stats(values, prefix):
    values = np.asarray(values, dtype=bool)
    count, total = int(values.sum()), len(values)
    if not total:
        return {prefix + "_count": 0, prefix + "_rate": math.nan,
                prefix + "_ci_low": math.nan, prefix + "_ci_high": math.nan}
    low = 0.0 if count == 0 else float(beta.ppf(0.025, count, total - count + 1))
    high = 1.0 if count == total else float(beta.ppf(0.975, count + 1, total - count))
    return {prefix + "_count": count, prefix + "_rate": count / total,
            prefix + "_ci_low": low, prefix + "_ci_high": high}


def quantiles(values, prefix):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {prefix + "_" + name: float(np.quantile(values, quantile)) if len(values) else math.nan
            for name, quantile in [("p10", .1), ("median", .5), ("p90", .9)]}


def paired_frequencies(cache, task, models):
    folder = cache / task
    example_path = folder / "examples.json"
    if not example_path.exists():
        raise FileNotFoundError(f"Missing {example_path}. --cache must point to Experiment 1's cache directory.")
    examples = read_json(example_path)
    if not examples or len({str(e['id']) for e in examples}) != len(examples):
        raise ValueError(f"Empty examples or duplicate IDs in {example_path}")
    expected = {str(e["id"]): e["gold"] for e in examples}
    data, hashes = {}, {"examples": hashlib.sha256(example_path.read_bytes()).hexdigest()}
    for model in models:
        path = folder / (slug(model) + ".jsonl")
        if not path.exists():
            raise FileNotFoundError(f"Missing predictions: {path}. Finish Experiment 1 evaluate for this model.")
        records = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            eid = str(r["id"])
            if eid in records:
                raise ValueError(f"Duplicate prediction ID {eid} in {path}")
            if type(r.get("correct")) is not bool:
                raise ValueError(f"Prediction correctness must be boolean in {path}")
            records[eid] = r
        if set(records) != set(expected):
            raise ValueError(f"Incomplete or mismatched predictions: {path}")
        if any(records[eid]["gold"] != gold for eid, gold in expected.items()):
            raise ValueError(f"Reference labels differ in {path}")
        data[model] = records
        hashes[model] = hashlib.sha256(path.read_bytes()).hexdigest()
    for ma, mb in itertools.combinations(models, 2):
        ca = np.array([data[ma][str(e["id"])]["correct"] for e in examples])
        cb = np.array([data[mb][str(e["id"])]["correct"] for e in examples])
        counts = np.array([(ca & ~cb).sum(), (~ca & cb).sum(), (ca == cb).sum()])
        f = counts / len(examples)
        summary = {"task": task, "model_a": ma, "model_b": mb,
                   "benchmark_examples": len(examples), "accuracy_a": ca.mean(), "accuracy_b": cb.mean(),
                   "a_only_count": int(counts[0]), "b_only_count": int(counts[1]), "tie_count": int(counts[2]),
                   "f_a_only": f[0], "f_b_only": f[1], "f_tie": f[2], "true_gap": f[0] - f[1],
                   "cache_pair_sha256": digest({k: hashes[k] for k in ["examples", ma, mb]})}
        yield f, summary

