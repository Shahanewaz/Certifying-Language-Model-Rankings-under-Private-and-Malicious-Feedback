#!/usr/bin/env python3
"""Experiment 3: report-adaptive poisoning and contamination-bound sensitivity.

Reuses Experiment 1 caches. simulate writes summary.csv and trials.csv.gz.
plot reads summary.csv only. All attacks respect valid report syntax.
"""
import argparse
from decimal import Decimal, ROUND_FLOOR
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import certificates as c

ATTACKS = ["fixed_favor_weaker", "random", "adaptive_cancel", "adaptive_error", "adaptive_abstain"]
METHODS = ["theorem1", "theorem3"]
NAMES = {"fixed_favor_weaker": "Fixed: favor weaker", "random": "Random reports",
         "adaptive_cancel": "Adaptive: cancel score", "adaptive_error": "Adaptive: seek error",
         "adaptive_abstain": "Adaptive: seek abstention"}


def floor_count(N, fraction):
    return int((Decimal(N)*Decimal(str(fraction))).to_integral_value(rounding=ROUND_FLOOR))


def evaluate_counts(counts, N, p, q, bbar, delta, channel):
    s = counts / N
    clo, chi, t = c.theorem1_bounds(s, N, p, q, bbar, delta)
    l1, u1, f1 = c.query_extrema(clo, chi)
    l3, u3, f3, radii = c.theorem3_bounds(s, N, p, q, bbar, delta, channel)
    return {"theorem1": (l1, u1, f1, c.decision(l1, u1, f1)),
            "theorem3": (l3, u3, f3, c.decision(l3, u3, f3))}, clo, chi, t, radii


def fixed_payload(repetitions, m, f):
    result = np.zeros((repetitions, 3), dtype=np.int64)
    result[:, 0 if f[0] <= f[1] else 1] = m
    return result


def cancel_payload(honest, m, channel, target_score=0):
    """Reach the nearest attainable signed total; remaining GRR reports are ties."""
    required = np.rint(target_score - (honest[:, 0]-honest[:, 1])).astype(np.int64)
    shift = np.clip(required, -m, m)
    result = np.zeros_like(honest)
    result[:, 0] = np.maximum(shift, 0)
    result[:, 1] = np.maximum(-shift, 0)
    if channel == "GRR":
        result[:, 2] = m - np.abs(shift)
    # OUE's remaining reports can be all zero, which is valid syntax.
    return result


def candidate_payloads(honest, m, channel, grid_resolution, p, q, bbar, delta, N, random_payload, f):
    """A declared finite candidate family, NOT exhaustive global attack optimization."""
    R = len(honest)
    yield "fixed_favor_weaker", fixed_payload(R, m, f)
    yield "random_realization", random_payload
    if channel == "GRR":
        g = grid_resolution
        for i in range(g+1):
            for j in range(g+1-i):
                a1, a2 = m*i//g, m*j//g
                yield f"grid_{i}_{j}", np.tile([a1, a2, m-a1-a2], (R, 1))
    else:
        levels = sorted({m*i//grid_resolution for i in range(grid_resolution+1)})
        for values in itertools.product(levels, repeat=3):
            yield "grid_" + "_".join(map(str, values)), np.tile(values, (R, 1))
    yield "cancel", cancel_payload(honest, m, channel)
    # Honest-dependent candidates near the direct certificate's decision thresholds.
    *_, radii = c.theorem3_bounds(honest/N, N, p, q, bbar, delta, channel)
    threshold = N*(bbar+radii["r_direct"])
    for signed_target in [-math.ceil(threshold)-1, math.ceil(threshold)+1]:
        payload = cancel_payload(honest, m, channel, signed_target)
        yield f"threshold_{signed_target}", payload
    if channel == "OUE":
        payload = cancel_payload(honest, m, channel)
        for tie_support in [0, m]:
            variant = payload.copy()
            variant[:, 2] = tie_support
            yield f"cancel_tie_{tie_support}", variant
            both = variant.copy()
            extra = np.minimum(m-both[:, 0], m-both[:, 1])
            both[:, 0] += extra
            both[:, 1] += extra
            yield f"cancel_both_tie_{tie_support}", both


def attack_score(result, truth, objective):
    lo, hi, feasible, d = result
    if objective == "adaptive_error":
        margin = -hi if truth > 0 else lo if truth < 0 else np.maximum(lo, -hi)
        # Empty sets cannot certify, so they are worst for an error-seeking attacker.
        return np.zeros(len(d), dtype=int), np.where(feasible, margin, -np.inf)
    # First prefer abstention. Within abstentions prefer a nonempty interval
    # containing zero over an empty interval, then maximize depth around zero.
    primary = np.where(d == 0, np.where(feasible, 2, 1), 0)
    depth = np.minimum(-lo, hi)
    return primary, np.where(feasible, depth, -np.inf)


def search_attacks(honest, m, channel, N, p, q, bbar, delta, f, grid_resolution, random_payload, targets, objectives):
    """Vectorized per-trial selection after observing honest supports.

    Optimize separately against each specified certificate; each selected payload
    is then evaluated by BOTH certificates. The true gap is oracle side information.
    """
    R = len(honest)
    truth = int(np.sign(f[0]-f[1]))
    best = {(objective, target): {"payload": np.zeros_like(honest), "candidate": np.full(R, "", dtype=object),
                                 "primary": np.full(R, -1), "secondary": np.full(R, -np.inf)}
            for objective in objectives for target in targets}
    count = 0
    for name, payload in candidate_payloads(honest, m, channel, grid_resolution, p, q, bbar, delta, N, random_payload, f):
        count += 1
        results, *_ = evaluate_counts(honest+payload, N, p, q, bbar, delta, channel)
        for (objective, target), saved in best.items():
            primary, secondary = attack_score(results[target], truth, objective)
            replace = ((primary > saved["primary"]) |
                       ((primary == saved["primary"]) & (secondary > saved["secondary"])))
            # Deterministic first-candidate tie breaking, including all-infeasible cases.
            replace |= saved["candidate"] == ""
            saved["payload"][replace] = payload[replace]
            saved["candidate"][replace] = name
            saved["primary"][replace] = primary[replace]
            saved["secondary"][replace] = secondary[replace]
    return best, count


def assert_valid_payload(payload, m, channel):
    if not np.issubdtype(payload.dtype, np.integer) or (payload < 0).any() or (payload > m).any():
        raise ValueError("Invalid malicious support counts.")
    if channel == "GRR" and not (payload.sum(axis=1) == m).all():
        raise ValueError("GRR malicious counts must sum to the malicious report count.")
    # For OUE, every integer vector in [0,m]^3 is realizable by m binary reports:
    # report r has bit i=1 iff r < payload_i. This is constructive, not relaxation.


def simulate(args):
    out = Path(args.out)
    if any(out.glob("*")) and not args.overwrite:
        raise ValueError(f"{out} is not empty. Choose a new --out or pass --overwrite.")
    # Validate all inputs before opening outputs; do not rerun model inference.
    pairs = [(f, metadata) for task in args.tasks
             for f, metadata in c.paired_frequencies(Path(args.cache), task, args.models)]
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k != "function"}
    config.update(version="experiment3-v1", script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  core_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),
                  python=c.platform.python_version(),
                  packages={k: c.importlib.metadata.version(k) for k in ["numpy", "pandas", "scipy", "matplotlib"]},
                  target="Uniform empirical distribution over cached NLP examples",
                  honest_sampling="IID with replacement; honest membership fixed before draws",
                  pairing="Same honest reports across attacks, bounds and methods within a base setting",
                  attacker_information="All honest aggregate support counts; N, channel, assumed bound, certificate; oracle true gap",
                  search_scope="Exact selection over the declared finite candidate family; NOT a global optimum over all valid reports",
                  certificate_use="Both methods evaluated separately at delta; no after-observation method selection",
                  interpretation="Reliability guarantees apply only where the actual malicious count is within the assumed bound")
    c.write_json(out/"configuration.json", config)
    c.atomic_csv([metadata for _, metadata in pairs], out/"model_accuracy.csv")
    trial_stream = gzip.open(out/"trials.csv.gz", "wt", encoding="utf-8", newline="")
    summary_stream = (out/"summary.csv").open("w", encoding="utf-8", newline="")
    header = True
    try:
        for f, meta in pairs:
            print(f"{meta['task']}: {meta['model_a']} vs {meta['model_b']}; gap={meta['true_gap']:+.6f}", flush=True)
            grid = list(itertools.product(args.channels, args.epsilons, args.report_counts, args.contamination))
            try:
                from tqdm import tqdm
                grid = tqdm(grid, desc=f"Attack {meta['task']}")
            except ImportError:
                pass
            for channel, eps, N, nominal_b in grid:
                m = floor_count(N, nominal_b)
                n = N-m
                p, q = c.channel_params(channel, eps)
                identity = {"task": meta["task"], "model_a": meta["model_a"], "model_b": meta["model_b"],
                            "cache_pair_sha256": meta["cache_pair_sha256"], "N": N, "channel": channel,
                            "epsilon": eps, "nominal_b": nominal_b, "seed": args.seed}
                base_id = c.digest(identity)[:20]
                # Honest sampling never depends on selected attacks or the assumed bound.
                rng = np.random.default_rng(int(c.digest(identity)[:16], 16))
                honest = c.draw_honest_counts(rng, n, f, p, q, channel, args.repetitions)
                random_rng = np.random.default_rng(int(c.digest([identity, "random_attack"])[:16], 16))
                random_payload = (random_rng.multinomial(m, [1/3]*3, size=args.repetitions) if channel == "GRR"
                                  else random_rng.binomial(m, .5, size=honest.shape))
                fixed = fixed_payload(args.repetitions, m, f)
                for factor in args.bbar_factors:
                    if nominal_b == 0 and factor != args.bbar_factors[0]:
                        continue  # Identical zero-bound settings do not need repetitions.
                    bbar = nominal_b*factor
                    if bbar >= 1:
                        raise ValueError("A nominal contamination times bound factor is >=1.")
                    valid = m <= floor_count(N, bbar)
                    fixed_results, *_ = evaluate_counts(honest+fixed, N, p, q, bbar, args.delta, channel)
                    active_objectives = [name for name in args.attacks if name in ["adaptive_error", "adaptive_abstain"]]
                    best, pool_count = ({}, 0) if not m or not active_objectives else search_attacks(
                        honest, m, channel, N, p, q, bbar, args.delta, f, args.grid_resolution,
                        random_payload, args.attack_targets, active_objectives)
                    scenarios = []
                    if not m:
                        scenarios.append(("none", "common", np.zeros_like(honest), np.full(args.repetitions, "no_attack"), 0))
                    else:
                        for attack in args.attacks:
                            if attack == "fixed_favor_weaker":
                                scenarios.append((attack, "common", fixed, np.full(args.repetitions, "fixed"), 1))
                            elif attack == "random":
                                scenarios.append((attack, "common", random_payload, np.full(args.repetitions, "random"), 1))
                            elif attack == "adaptive_cancel":
                                scenarios.append((attack, "common", cancel_payload(honest, m, channel), np.full(args.repetitions, "cancel"), 1))
                            else:
                                for target in args.attack_targets:
                                    saved = best[(attack, target)]
                                    scenarios.append((attack, target, saved["payload"], saved["candidate"], pool_count))
                    for attack, target, payload, chosen, candidate_count in scenarios:
                        assert_valid_payload(payload, m, channel)
                        observed = honest+payload
                        results, clo, chi, t, radii = evaluate_counts(observed, N, p, q, bbar, args.delta, channel)
                        setting_id = c.digest([identity, bbar, factor, args.delta, attack, target, args.grid_resolution])[:20]
                        base = {**meta, "base_id": base_id, "setting_id": setting_id, "channel": channel, "epsilon": eps,
                                "N": N, "n": n, "malicious_count": m, "nominal_b": nominal_b, "actual_b": m/N,
                                "bbar": bbar, "bbar_factor": factor, "assumption_satisfied": valid,
                                "bound_regime": "valid" if valid else "underestimated",
                                "attack": attack, "attack_target": target, "is_adaptive": attack.startswith("adaptive"),
                                "candidate_count": candidate_count, "grid_resolution": args.grid_resolution,
                                "search_scope": "finite_candidate_selection" if attack in active_objectives else "specified_strategy",
                                "delta": args.delta, "repetitions": args.repetitions, "seed": args.seed,
                                "p": p, "q": q, "t_coordinate": t, **radii}
                        truth = int(np.sign(meta["true_gap"]))
                        full_covered = (f >= clo).all(axis=1) & (f <= chi).all(axis=1)
                        trial = {"setting_id": setting_id, "base_id": base_id, "trial": np.arange(args.repetitions),
                                 **{key: base[key] for key in ["task", "model_a", "model_b", "channel", "epsilon", "N", "n",
                                                             "malicious_count", "nominal_b", "actual_b", "bbar", "bbar_factor",
                                                             "assumption_satisfied", "attack", "attack_target", "true_gap", "delta"]},
                                 "selected_candidate": chosen, "theorem1_distribution_covered": full_covered}
                        for i in range(3):
                            trial[f"honest_support_{i+1}"] = honest[:, i]
                            trial[f"malicious_support_{i+1}"] = payload[:, i]
                            trial[f"observed_support_{i+1}"] = observed[:, i]
                            trial[f"theorem1_coordinate_lower_{i+1}"] = clo[:, i]
                            trial[f"theorem1_coordinate_upper_{i+1}"] = chi[:, i]
                        rows = []
                        for method in METHODS:
                            lo, hi, feasible, d = results[method]
                            wrong = (d != 0) & (d != truth)
                            correct = (d != 0) & (d == truth)
                            gap_covered = feasible & (lo <= meta["true_gap"]) & (meta["true_gap"] <= hi)
                            fixed_d = fixed_results[method][3]
                            fixed_wrong = (fixed_d != 0) & (fixed_d != truth)
                            row = {**base, "method": method, "feasible_trials": int(feasible.sum()),
                                   **c.rate_stats(gap_covered, "gap_coverage"), **c.rate_stats(wrong, "error"),
                                   **c.rate_stats(correct, "correct_certification"), **c.rate_stats(d != 0, "certification"),
                                   **c.rate_stats(d == 0, "abstention"), **c.rate_stats(~feasible, "empty_interval"),
                                   **c.quantiles((hi-lo)[feasible], "gap_width"),
                                   "paired_error_gain_vs_fixed": float(wrong.mean()-fixed_wrong.mean()),
                                   "paired_abstention_gain_vs_fixed": float((d == 0).mean()-(fixed_d == 0).mean())}
                            row.update(c.rate_stats(full_covered, "distribution_coverage") if method == "theorem1" else
                                       {"distribution_coverage_"+k: math.nan for k in ["count", "rate", "ci_low", "ci_high"]})
                            rows.append(row)
                            trial.update({method+"_lower": lo, method+"_upper": hi, method+"_feasible": feasible,
                                          method+"_decision": d, method+"_error": wrong, method+"_gap_covered": gap_covered,
                                          method+"_correct_certification": correct, method+"_fixed_decision": fixed_d})
                        pd.DataFrame(trial).to_csv(trial_stream, header=header, index=False)
                        pd.DataFrame(rows).to_csv(summary_stream, header=header, index=False)
                        summary_stream.flush()
                        header = False
                trial_stream.flush()
    finally:
        trial_stream.close()
        summary_stream.close()
    print(f"Saved {out/'summary.csv'} and {out/'trials.csv.gz'}. Underestimated-bound settings are labeled, not guaranteed.")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(args.csv)
    required = {"setting_id", "method", "task", "model_a", "model_b", "channel", "epsilon", "N", "nominal_b",
                "bbar", "bbar_factor", "attack", "attack_target", "assumption_satisfied", "delta", "true_gap"}
    for metric in ["gap_coverage", "error", "abstention", "correct_certification"]:
        required.update(metric+suffix for suffix in ["_rate", "_ci_low", "_ci_high"])
    if required-set(df):
        raise ValueError("Not an Experiment 3 summary CSV; missing " + ", ".join(sorted(required-set(df))))
    if df.duplicated(["setting_id", "method"]).any():
        raise ValueError("Duplicate setting/method rows; use one summary.csv.")
    for key, col in [("tasks", "task"), ("channels", "channel"), ("epsilons", "epsilon"), ("contamination", "nominal_b")]:
        values = getattr(args, key)
        if values:
            if col in ["epsilon", "nominal_b"]:
                df = df[np.logical_or.reduce([np.isclose(df[col], value, rtol=0, atol=1e-12) for value in values])]
            else:
                df = df[df[col].isin(values)]
    if df.empty:
        raise ValueError("No rows match the plot filters.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    palette = dict(zip(["none"]+ATTACKS, plt.cm.tab10(np.arange(6))))
    figure_index = []

    def save(fig, stem, metadata):
        fig.savefig(out/(stem+".pdf"), bbox_inches="tight")
        fig.savefig(out/(stem+".png"), dpi=200, bbox_inches="tight")
        figure_index.append({"figure_stem": stem, "source_csv": str(Path(args.csv)), **metadata})
        plt.close(fig)

    for target in args.attack_targets:
        data = df[df.attack_target.isin(["common", target])].copy()
        groups = ["task", "model_a", "model_b", "channel", "epsilon", "nominal_b", "bbar_factor", "delta"]
        for key, sub in data.groupby(groups):
            task, ma, mb, channel, eps, nominal_b, factor, delta = key
            bbar = sub.bbar.iloc[0]
            valid = sub.assumption_satisfied.astype(str).str.lower().isin(["true", "1"]).all()
            title = f"{task.upper()} | {ma} vs {mb}\n{channel}, epsilon={eps:g}, b={nominal_b:g}, bound={bbar:g} | adaptive target: {target}"
            fig, axes = plt.subplots(2, 4, figsize=(15, 6.8), squeeze=False)
            metrics = ["gap_coverage", "error", "abstention", "correct_certification"]
            for row, method in enumerate(METHODS):
                for col, metric in enumerate(metrics):
                    ax = axes[row, col]
                    for attack, curve in sub[sub.method == method].groupby("attack"):
                        curve = curve.sort_values("N")
                        ax.plot(curve.N, curve[metric+"_rate"], color=palette[attack], marker="o", ms=3,
                                label=NAMES.get(attack, "No malicious reports"))
                        ax.fill_between(curve.N.to_numpy(), curve[metric+"_ci_low"].to_numpy(),
                                        curve[metric+"_ci_high"].to_numpy(), color=palette[attack], alpha=.09)
                    if metric in ["gap_coverage", "error"]:
                        ax.axhline(1-delta if metric == "gap_coverage" else delta, color="black", ls="--", lw=.8)
                    ax.set_xscale("log")
                    ax.set_ylim(-.015, 1.015)
                    ax.grid(alpha=.2)
                    ax.set_title(metric.replace("_", " ").title(), fontsize=9)
                    ax.set_xlabel("Total reports N")
                    if col == 0:
                        ax.set_ylabel(method + "\nRate")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .88), ncol=3, fontsize=8)
            fig.suptitle(title, fontsize=10)
            note = "Bound satisfied at all displayed N." if valid else "Some actual counts exceed the assumed bound: theorem guarantees do not apply there."
            fig.text(.01, .012, note + " Shading: pointwise 95% Monte Carlo intervals. Adaptive search is over a declared finite candidate family.", fontsize=7)
            fig.tight_layout(rect=(0, .05, 1, .77))
            stem = f"{task}_{channel}_eps{eps:g}_b{nominal_b:g}_factor{factor:g}_{target}_{c.digest(list(key))[:8]}"
            save(fig, stem+"_attack_curves", {**dict(zip(groups, key)), "attack_target": target, "figure_type": "attack_curves"})

        # Heatmaps use one tested N; do not average budgets or bound regimes.
        N = args.heatmap_N or int(data.N.max())
        largest = data[data.N == N]
        groups = ["task", "model_a", "model_b", "channel", "epsilon", "attack", "delta"]
        for key, sub in largest.groupby(groups):
            task, ma, mb, channel, eps, attack, delta = key
            fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), squeeze=False)
            for row, method in enumerate(METHODS):
                method_data = sub[sub.method == method]
                for col, metric in enumerate(["error", "abstention"]):
                    heat = method_data.pivot(index="bbar_factor", columns="nominal_b", values=metric+"_rate").sort_index().sort_index(axis=1)
                    validity = method_data.assign(valid=method_data.assumption_satisfied.astype(str).str.lower().isin(["true", "1"])).pivot(
                        index="bbar_factor", columns="nominal_b", values="valid").reindex(index=heat.index, columns=heat.columns)
                    ax = axes[row, col]
                    im = ax.imshow(np.ma.masked_invalid(heat.to_numpy()), origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
                    ax.set_xticks(range(len(heat.columns)), [f"{b*100:g}%" for b in heat.columns])
                    ax.set_yticks(range(len(heat.index)), [f"{v:g}" for v in heat.index])
                    for y in range(len(heat.index)):
                        for x in range(len(heat.columns)):
                            if pd.notna(heat.iloc[y, x]):
                                val = heat.iloc[y, x]
                                ax.text(x, y, f"{val:.2f}" + ("*" if not bool(validity.iloc[y, x]) else ""),
                                        ha="center", va="center", fontsize=8, color="white" if val < .65 else "black")
                    ax.set_xlabel("Nominal malicious fraction")
                    ax.set_ylabel("Assumed bound / nominal fraction")
                    ax.set_title(f"{method}: {metric} rate", fontsize=10)
                    fig.colorbar(im, ax=ax, fraction=.046)
            fig.suptitle(f"{task.upper()} | {ma} vs {mb}\n{channel}, epsilon={eps:g}, N={N:,} | {attack} | adaptive target: {target}", fontsize=10)
            fig.text(.01, .01, "* Actual malicious count exceeds assumed bound; no theorem guarantee. Blank cells were not evaluated. Rates are empirical.", fontsize=7)
            fig.tight_layout(rect=(0, .05, 1, .88))
            stem = f"{task}_{channel}_eps{eps:g}_{attack}_{target}_N{N}_{c.digest(list(key))[:8]}_bound_sensitivity"
            save(fig, stem, {**dict(zip(groups, key)), "attack_target": target, "N": N, "figure_type": "bound_sensitivity"})
    selected_N = args.heatmap_N or int(df.N.max())
    df[df.N == selected_N].to_csv(out/"attack_summary_at_selected_N.csv", index=False)
    pd.DataFrame(figure_index).to_csv(out/"figure_index.csv", index=False)
    print(f"Saved figures and attack_summary_at_selected_N.csv in {out}. Plotting used only the CSV.")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--cache", required=True)
    s.add_argument("--out", default="experiment3_results")
    s.add_argument("--tasks", nargs="+", choices=["snli", "sst2", "boolq"], default=["snli"])
    s.add_argument("--models", nargs="+", default=c.MODELS)
    s.add_argument("--channels", nargs="+", choices=["GRR", "OUE"], default=["GRR", "OUE"])
    s.add_argument("--epsilons", nargs="+", type=float, default=[.1, .5, 1, 2, 5])
    s.add_argument("--report-counts", nargs="+", type=int, default=[1000, 10000, 100000, 1000000])
    s.add_argument("--contamination", nargs="+", type=float, default=[0, .001, .01, .05])
    s.add_argument("--bbar-factors", nargs="+", type=float, default=[.5, 1, 2])
    s.add_argument("--attacks", nargs="+", choices=ATTACKS, default=ATTACKS)
    s.add_argument("--attack-targets", nargs="+", choices=METHODS, default=METHODS)
    s.add_argument("--grid-resolution", type=int, default=2)
    s.add_argument("--delta", type=float, default=.05)
    s.add_argument("--repetitions", type=int, default=1000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(function=simulate)
    t = sub.add_parser("plot")
    t.add_argument("--csv", required=True)
    t.add_argument("--out", default="experiment3_figures")
    t.add_argument("--tasks", nargs="+", choices=["snli", "sst2", "boolq"])
    t.add_argument("--channels", nargs="+", choices=["GRR", "OUE"])
    t.add_argument("--epsilons", nargs="+", type=float)
    t.add_argument("--contamination", nargs="+", type=float)
    t.add_argument("--attack-targets", nargs="+", choices=METHODS, default=["theorem1"])
    t.add_argument("--heatmap-N", type=int, default=None, help="Default: largest tested N after filters")
    t.set_defaults(function=plot)
    return p


def main():
    p = parser()
    args = p.parse_args()
    for key in ["tasks", "models", "channels", "epsilons", "report_counts", "contamination", "bbar_factors", "attacks", "attack_targets"]:
        values = getattr(args, key, None)
        if values is not None and len(values) != len(set(values)):
            p.error("Duplicate values in --"+key.replace("_", "-"))
    if args.command == "simulate":
        if len(args.models) < 2 or args.repetitions < 1 or args.seed < 0 or args.grid_resolution < 1:
            p.error("Require >=2 models, positive repetitions/grid resolution and a nonnegative seed.")
        if not math.isfinite(args.delta) or not 0 < args.delta < 1:
            p.error("delta must be in (0,1).")
        if any(N < 1 for N in args.report_counts):
            p.error("Report counts must be positive.")
        if any(not math.isfinite(b) or not 0 <= b < 1 for b in args.contamination):
            p.error("Contamination must be in [0,1).")
        if any(not math.isfinite(v) or v < 0 for v in args.bbar_factors):
            p.error("Bound factors must be finite and nonnegative.")
        if max(args.contamination)*max(args.bbar_factors) >= 1:
            p.error("Each nominal contamination times bound factor must be <1.")
        for eps in args.epsilons:
            c.channel_params("GRR", eps)
    try:
        args.function(args)
    except (ValueError, FileNotFoundError) as exc:
        p.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
