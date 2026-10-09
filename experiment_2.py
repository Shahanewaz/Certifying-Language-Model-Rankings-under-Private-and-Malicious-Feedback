#!/usr/bin/env python3
"""Experiment 2: compare Theorem 1 and Theorem 3 on identical private reports.

Uses Experiment 1 prediction caches. No Ollama calls or dataset downloads.
simulate -> summary.csv + trials.csv.gz; plot reads summary.csv only.
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


def simulate(args):
    out = Path(args.out)
    if any(out.glob("*")) and not args.overwrite:
        raise ValueError(f"{out} is not empty. Choose a new --out or pass --overwrite to replace Experiment 2 outputs.")
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k != "function"}
    config.update(version=VERSION, python=platform.python_version(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  target="Uniform empirical distribution over Experiment 1 cached examples",
                  sampling="IID with replacement; exact aggregate channel simulation",
                  pairing="Both methods use the SAME honest and malicious counts in each trial",
                  guarantee="Separate method evaluation, each at delta; no adaptive selection or intersection",
                  attack_scope="Fixed valid attacks; adaptive attacks are not implemented in Experiment 2")
    config["packages"] = {k: importlib.metadata.version(k) for k in ["numpy", "pandas", "scipy", "matplotlib"]}
    write_json(out / "configuration.json", config)
    model_rows = []
    stream = gzip.open(out / "trials.csv.gz", "wt", encoding="utf-8", newline="")
    summary_stream = (out / "summary.csv").open("w", encoding="utf-8", newline="")
    header = True
    try:
        for task in args.tasks:
            for f, model_summary in paired_frequencies(Path(args.cache), task, args.models):
                model_rows.append(model_summary)
                print(f"{task}: {model_summary['model_a']} vs {model_summary['model_b']}; "
                      f"empirical gap={model_summary['true_gap']:+.6f}", flush=True)
                grid = list(itertools.product(args.channels, args.epsilons, args.report_counts,
                                              args.contamination, args.attacks))
                try:
                    from tqdm import tqdm
                    grid = tqdm(grid, desc=f"Compare {task}")
                except ImportError:
                    pass
                for channel, eps, N, nominal_b, attack in grid:
                    m = math.floor(N * nominal_b + 1e-9)
                    if m == 0 and attack != args.attacks[0]:
                        continue
                    n = N - m
                    bbar = nominal_b if args.bbar is None else args.bbar
                    valid = m <= math.floor(N * bbar + 1e-9)
                    p, q = channel_params(channel, eps)
                    identity = {"task": task, "model_a": model_summary["model_a"], "model_b": model_summary["model_b"],
                                "cache_pair_sha256": model_summary["cache_pair_sha256"], "channel": channel,
                                "epsilon": eps, "N": N, "nominal_b": nominal_b, "bbar": bbar, "attack": attack,
                                "seed": args.seed, "delta": args.delta}
                    setting_id = digest(identity)[:20]
                    rng = np.random.default_rng(int(digest(identity)[:16], 16))
                    honest = draw_honest_counts(rng, n, f, p, q, channel, args.repetitions)
                    malicious = np.zeros_like(honest)
                    if m:
                        if attack == "random":
                            malicious = (rng.multinomial(m, [1/3]*3, size=args.repetitions) if channel == "GRR"
                                         else rng.binomial(m, .5, size=honest.shape))
                        else:
                            target = (0 if f[0] <= f[1] else 1) if attack == "favor_weaker" else (0 if attack == "favor_a" else 1)
                            malicious[:, target] = m
                    observed = honest + malicious
                    s = observed / N
                    clo, chi, t = theorem1_bounds(s, N, p, q, bbar, args.delta)
                    l1, u1, feasible1 = query_extrema(clo, chi)
                    l3, u3, feasible3, radii = theorem3_bounds(s, N, p, q, bbar, args.delta, channel)
                    d1, d3 = decision(l1, u1, feasible1), decision(l3, u3, feasible3)
                    truth = int(np.sign(model_summary["true_gap"]))
                    correct1, correct3 = (d1 != 0) & (d1 == truth), (d3 != 0) & (d3 == truth)
                    common = feasible1 & feasible3
                    w1, w3 = u1 - l1, u3 - l3
                    ratios = np.divide(w3[common], w1[common], out=np.full(common.sum(), np.nan), where=w1[common] > 0)
                    paired = {"common_feasible_trials": int(common.sum()),
                              "paired_correct_certification_gain": float(correct3.mean() - correct1.mean()),
                              "paired_certification_gain": float((d3 != 0).mean() - (d1 != 0).mean()),
                              "t3_only_correct_count": int((correct3 & ~correct1).sum()),
                              "t1_only_correct_count": int((correct1 & ~correct3).sum()),
                              "both_correct_count": int((correct1 & correct3).sum()),
                              "neither_correct_count": int((~correct1 & ~correct3).sum()),
                              **quantiles(ratios, "paired_width_ratio"),
                              **quantiles((w3 - w1)[common], "paired_width_difference")}
                    base = {**model_summary, "setting_id": setting_id, "channel": channel, "epsilon": eps,
                            "N": N, "n": n, "malicious_count": m, "nominal_b": nominal_b, "actual_b": m/N,
                            "bbar": bbar, "assumption_satisfied": valid, "attack": attack if m else "none",
                            "requested_attack": attack, "delta": args.delta, "repetitions": args.repetitions,
                            "seed": args.seed, "p": p, "q": q, "t_coordinate": t, **radii, **paired}
                    gap = model_summary["true_gap"]
                    full_coverage = feasible1 & (f >= clo).all(axis=1) & (f <= chi).all(axis=1)
                    trial = {"setting_id": setting_id, "trial": np.arange(args.repetitions),
                             **{k: base[k] for k in ["task", "model_a", "model_b", "channel", "epsilon", "N", "n",
                                                     "malicious_count", "nominal_b", "actual_b", "bbar", "attack", "delta",
                                                     "true_gap", "assumption_satisfied"]},
                             "theorem1_distribution_covered": full_coverage}
                    for i in range(3):
                        trial[f"honest_support_{i+1}"] = honest[:, i]
                        trial[f"malicious_support_{i+1}"] = malicious[:, i]
                        trial[f"observed_support_{i+1}"] = observed[:, i]
                        trial[f"theorem1_coordinate_lower_{i+1}"] = clo[:, i]
                        trial[f"theorem1_coordinate_upper_{i+1}"] = chi[:, i]
                    setting_rows = []
                    for method, lo, hi, feasible, d in [("theorem1", l1, u1, feasible1, d1), ("theorem3", l3, u3, feasible3, d3)]:
                        covered = feasible & (lo <= gap) & (gap <= hi)
                        wrong = (d != 0) & (d != truth)
                        correct = (d != 0) & (d == truth)
                        row = {**base, "method": method, "feasible_trials": int(feasible.sum()),
                               **quantiles(lo[feasible], "gap_lower"), **quantiles(hi[feasible], "gap_upper"),
                               **quantiles((hi-lo)[feasible], "gap_width"),
                               **rate_stats(covered, "gap_coverage"), **rate_stats(wrong, "error"),
                               **rate_stats(correct, "correct_certification"), **rate_stats(d != 0, "certification"),
                               **rate_stats(d == 0, "abstention"), **rate_stats(~feasible, "empty_interval")}
                        if method == "theorem1":
                            row.update(rate_stats(full_coverage, "distribution_coverage"))
                        else:
                            row.update({"distribution_coverage_" + k: math.nan for k in ["count", "rate", "ci_low", "ci_high"]})
                        setting_rows.append(row)
                        trial.update({method + "_lower": lo, method + "_upper": hi,
                                      method + "_feasible": feasible, method + "_decision": d,
                                      method + "_gap_covered": covered, method + "_error": wrong,
                                      method + "_correct_certification": correct})
                    pd.DataFrame(trial).to_csv(stream, index=False, header=header)
                    pd.DataFrame(setting_rows).to_csv(summary_stream, index=False, header=header)
                    summary_stream.flush()
                    header = False
                stream.flush()
                atomic_csv(model_rows, out / "model_accuracy.csv")
    finally:
        stream.close()
        summary_stream.close()
    print(f"Saved {out / 'summary.csv'} and {out / 'trials.csv.gz'}")
    print("Trial decisions: +1=A better, -1=B better, 0=abstain. Each method runs separately at delta.")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(args.csv)
    required = {"setting_id", "task", "model_a", "model_b", "method", "channel", "epsilon", "N", "nominal_b", "bbar",
                "attack", "delta", "true_gap", "assumption_satisfied", "correct_certification_rate",
                "correct_certification_ci_low", "correct_certification_ci_high", "gap_width_median", "gap_width_p10",
                "gap_width_p90", "gap_coverage_rate", "gap_coverage_ci_low", "gap_coverage_ci_high",
                "error_rate", "error_ci_low", "error_ci_high", "paired_correct_certification_gain",
                "paired_width_ratio_median", "common_feasible_trials"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Not an Experiment 2 summary CSV; missing: " + ", ".join(sorted(missing)))
    if df.duplicated(["setting_id", "method"]).any():
        raise ValueError("Duplicate setting/method rows; use one run's summary.csv.")
    if not set(df.method.unique()).issubset(METHODS):
        raise ValueError("Unrecognized certificate method.")
    if not (df.groupby("setting_id").method.nunique() == 2).all():
        raise ValueError("Each setting must contain both theorem1 and theorem3 summary rows.")
    for key in ["tasks", "channels", "epsilons", "contamination"]:
        selected = getattr(args, key)
        if selected:
            col = {"tasks": "task", "channels": "channel", "epsilons": "epsilon", "contamination": "nominal_b"}[key]
            if col in ["epsilon", "nominal_b"]:
                df = df[np.logical_or.reduce([np.isclose(df[col], value, rtol=0, atol=1e-12) for value in selected])]
            else:
                df = df[df[col].isin(selected)]
    # Zero-malicious-count settings have attack='none' and remain in each selected attack panel.
    if args.attacks:
        df = df[df.attack.isin(args.attacks + ["none"])]
    if df.empty:
        raise ValueError("No rows match the plot filters.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    colors = {"theorem1": "#2458a6", "theorem3": "#d46b19"}
    styles = {"theorem1": "o", "theorem3": "s"}
    groups = ["task", "model_a", "model_b", "channel", "nominal_b", "bbar", "delta", "requested_attack"]
    if "requested_attack" not in df:
        raise ValueError("Missing requested_attack column; use an unmodified Experiment 2 summary CSV.")
    budgets = []
    for key, sub in df.groupby(groups, sort=True, dropna=False):
        task, ma, mb, channel, nominal_b, bbar, delta, attack = key
        epsilons = sorted(sub.epsilon.unique())
        title = f"{task.upper()} | {ma} vs {mb}\n{channel} | nominal b={nominal_b:g}, bound={bbar:g} | {attack} | gap={sub.true_gap.iloc[0]:+.4f}"
        valid = sub.assumption_satisfied.astype(str).str.lower().isin(["true", "1"]).all()
        note = "" if valid else "ASSUMPTION VIOLATED: actual contamination exceeds the stated bound. "
        stem = f"{task}_{slug(ma)}_vs_{slug(mb)}_{channel}_b{nominal_b:g}_{attack}_" + digest(list(key))[:8]

        def save(fig, name):
            fig.savefig(out / (stem + "_" + name + ".pdf"), bbox_inches="tight")
            fig.savefig(out / (stem + "_" + name + ".png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

        for metric, kind in [("correct_certification", "certification"), ("gap_width", "width")]:
            fig, axes = plt.subplots(1, len(epsilons), figsize=(3.2*len(epsilons), 3.8), squeeze=False, sharey=True)
            for ax, eps in zip(axes[0], epsilons):
                data = sub[sub.epsilon == eps]
                for method in METHODS:
                    curve = data[data.method == method].sort_values("N")
                    if curve.empty:
                        continue
                    y = curve[metric + ("_rate" if kind == "certification" else "_median")]
                    low = curve[metric + ("_ci_low" if kind == "certification" else "_p10")]
                    high = curve[metric + ("_ci_high" if kind == "certification" else "_p90")]
                    ax.plot(curve.N, y, marker=styles[method], ms=3, color=colors[method], label=LABELS[method])
                    ax.fill_between(curve.N.to_numpy(), low.to_numpy(), high.to_numpy(), color=colors[method], alpha=.13)
                if kind == "certification":
                    ax.axhline(args.target_rate, color="gray", ls=":", lw=.8)
                    ax.set_ylim(-.02, 1.02)
                    if data.correct_certification_rate.max() == 0:
                        ax.text(.5, .15, "No correct certifications observed", transform=ax.transAxes, ha="center", fontsize=7)
                ax.set_xscale("log")
                ax.set_title(f"epsilon={eps:g}")
                ax.set_xlabel("Total reports N")
                ax.grid(alpha=.2)
            axes[0, 0].set_ylabel("Correct certification rate" if kind == "certification" else "Gap interval width")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .87), ncol=2, fontsize=8)
            fig.suptitle(title, fontsize=10)
            fig.text(.02, .015, note + ("Shading: pointwise 95% Monte Carlo rate intervals." if kind == "certification"
                                       else "Shading: 10th-90th width percentiles among each method's nonempty intervals; descriptive, not coverage."), fontsize=7)
            fig.tight_layout(rect=(0, .07, 1, .80))
            save(fig, kind)

        fig, axes = plt.subplots(2, len(epsilons), figsize=(3.2*len(epsilons), 6.0), squeeze=False, sharex=True, sharey="row")
        for col, eps in enumerate(epsilons):
            for method in METHODS:
                curve = sub[(sub.epsilon == eps) & (sub.method == method)].sort_values("N")
                for row, metric in enumerate(["gap_coverage", "error"]):
                    ax = axes[row, col]
                    ax.plot(curve.N, curve[metric+"_rate"], marker=styles[method], ms=3, color=colors[method], label=LABELS[method])
                    ax.fill_between(curve.N.to_numpy(), curve[metric+"_ci_low"].to_numpy(), curve[metric+"_ci_high"].to_numpy(), color=colors[method], alpha=.13)
                    ax.set_xscale("log")
                    ax.grid(alpha=.2)
            axes[0, col].axhline(1-delta, color="black", ls="--", lw=.8)
            axes[0, col].set_title(f"epsilon={eps:g}")
            axes[0, col].set_ylim(0, 1.02)
            axes[1, col].axhline(delta, color="black", ls="--", lw=.8)
            axes[1, col].set_ylim(0, max(.08, 1.15*sub.error_ci_high.max()))
            axes[1, col].set_xlabel("Total reports N")
        axes[0, 0].set_ylabel("Gap coverage rate")
        axes[1, 0].set_ylabel("Incorrect certification rate")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .89), ncol=2, fontsize=8)
        fig.suptitle(title, fontsize=10)
        fig.text(.02, .01, note + "Dashed lines: 1-delta and delta. Shading: pointwise 95% Monte Carlo intervals; not simultaneous guarantees.", fontsize=7)
        fig.tight_layout(rect=(0, .05, 1, .83))
        save(fig, "coverage_error")

        paired = sub.drop_duplicates("setting_id")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, field, label in zip(axes, ["paired_correct_certification_gain", "paired_width_ratio_median"],
                                    ["Correct certification gain: Thm 3 - Thm 1", "Median paired width ratio: Thm 3 / Thm 1"]):
            heat = paired.pivot(index="epsilon", columns="N", values=field).sort_index().sort_index(axis=1)
            values = np.ma.masked_invalid(heat.to_numpy())
            if field.endswith("gain"):
                bound = max(.01, float(np.nanmax(np.abs(values))) if values.count() else .01)
                im = ax.imshow(values, aspect="auto", origin="lower", cmap="RdBu", vmin=-bound, vmax=bound)
            else:
                im = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(len(heat.columns)), [f"{int(n):,}" for n in heat.columns], rotation=45, ha="right")
            ax.set_yticks(range(len(heat.index)), [f"{e:g}" for e in heat.index])
            ax.set_xlabel("Total reports N")
            ax.set_ylabel("epsilon")
            ax.set_title(label, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=.046)
        fig.suptitle(title, fontsize=10)
        fig.text(.02, .01, note + "Gain > 0 favors Thm 3. Ratio < 1 favors Thm 3; ratios use trials where BOTH intervals are nonempty. Blank = undefined.", fontsize=7)
        fig.tight_layout(rect=(0, .07, 1, .85))
        save(fig, "paired_comparison")

        for (eps, method), curve in sub.groupby(["epsilon", "method"]):
            curve = curve.sort_values("N")
            hits = curve[curve.correct_certification_ci_low >= args.target_rate]
            budgets.append(dict(zip(groups, key), epsilon=eps, method=method, target_rate=args.target_rate,
                                criterion="pointwise 95% Monte Carlo lower bound >= target",
                                first_tested_N=int(hits.N.iloc[0]) if len(hits) else math.nan,
                                status="reached" if len(hits) else "not reached",
                                largest_tested_N=int(curve.N.max()), assumption_satisfied=valid))
    atomic_csv(budgets, out / "reports_required.csv")
    print(f"Saved figures and reports_required.csv in {out}. No simulation or Ollama calls were made.")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("simulate", help="Read Experiment 1 caches; compare certificates using identical trials")
    s.add_argument("--cache", required=True, help="Experiment 1 cache folder, e.g. three_models_three_tasks/cache")
    s.add_argument("--out", default="experiment2_results")
    s.add_argument("--tasks", nargs="+", choices=["snli", "sst2", "boolq"], default=["snli"])
    s.add_argument("--models", nargs="+", default=MODELS)
    s.add_argument("--channels", nargs="+", choices=["GRR", "OUE"], default=["GRR", "OUE"])
    s.add_argument("--epsilons", nargs="+", type=float, default=EPSILONS)
    s.add_argument("--report-counts", nargs="+", type=int, default=[500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000])
    s.add_argument("--contamination", nargs="+", type=float, default=[0, .001, .002, .005, .01, .05, .1])
    s.add_argument("--bbar", type=float, default=None, help="Default: nominal contamination in each setting")
    s.add_argument("--attacks", nargs="+", choices=["favor_weaker", "favor_a", "favor_b", "random"], default=["favor_weaker", "random"])
    s.add_argument("--delta", type=float, default=.05)
    s.add_argument("--repetitions", type=int, default=1000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(function=simulate)
    t = sub.add_parser("plot", help="Read a saved summary.csv; no rerun of models or simulation")
    t.add_argument("--csv", required=True, help="Experiment 2 summary.csv")
    t.add_argument("--out", default="experiment2_figures")
    t.add_argument("--tasks", nargs="+", choices=["snli", "sst2", "boolq"])
    t.add_argument("--channels", nargs="+", choices=["GRR", "OUE"])
    t.add_argument("--epsilons", nargs="+", type=float)
    t.add_argument("--contamination", nargs="+", type=float)
    t.add_argument("--attacks", nargs="+", choices=["favor_weaker", "favor_a", "favor_b", "random"], default=["favor_weaker"])
    t.add_argument("--target-rate", type=float, default=.9)
    t.set_defaults(function=plot)
    return p


def main():
    p = parser()
    args = p.parse_args()
    for name in ["tasks", "models", "channels", "epsilons", "report_counts", "contamination", "attacks"]:
        values = getattr(args, name, None)
        if values is not None and len(values) != len(set(values)):
            p.error(f"Duplicate values in --{name.replace('_', '-')}")
    if args.command == "simulate":
        if len(args.models) < 2:
            p.error("Supply at least two models.")
        if not math.isfinite(args.delta) or not 0 < args.delta < 1 or args.repetitions < 1:
            p.error("Require 0 < delta < 1 and positive repetitions.")
        if args.seed < 0 or any(n < 1 for n in args.report_counts):
            p.error("Seed must be nonnegative and report counts positive.")
        if any(not math.isfinite(b) or not 0 <= b < 1 for b in args.contamination):
            p.error("Contamination must be finite and in [0,1).")
        if args.bbar is not None and (not math.isfinite(args.bbar) or not 0 <= args.bbar < 1):
            p.error("bbar must be finite and in [0,1).")
        for epsilon in args.epsilons:
            channel_params("GRR", epsilon)
    elif not math.isfinite(args.target_rate) or not 0 < args.target_rate < 1:
        p.error("target-rate must be in (0,1).")
    try:
        args.function(args)
    except (ValueError, FileNotFoundError) as exc:
        p.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
