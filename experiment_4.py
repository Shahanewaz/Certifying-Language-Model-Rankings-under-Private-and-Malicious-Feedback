#!/usr/bin/env python3
"""Experiment 4: certified NLP model rankings within predefined linguistic strata.

Reuses Experiment 1 predictions and input texts from cached prompts. No new
inference or downloads. simulate writes CSVs; plot reads summary.csv only.
"""
import argparse
from decimal import Decimal, ROUND_FLOOR
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import certificates as c

METHODS = ["theorem1", "theorem3"]
NEGATION = {"no", "not", "never", "neither", "nor", "none", "nobody", "nothing", "nowhere", "without"}
CONTRAST = {"but", "however", "although", "though", "yet", "nevertheless"}
STOPWORDS = set("a an the and or of to in on at for with by from is are was were be been being do does did have has had this that these those it its as than then there here i you he she they we can could would will should may might must shall".split())
WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


def tokens(text):
    return WORD.findall(text.lower().replace("\u2019", "'"))


def negated(words):
    return any(word in NEGATION or word.endswith("n't") for word in words)


def overlap(reference_words, query_words):
    reference = set(reference_words)-STOPWORDS
    query = set(query_words)-STOPWORDS
    return len(reference & query)/len(query) if query else 0.0


def extract_text(example, task):
    keys = {"snli": ["premise", "hypothesis"], "sst2": ["review"], "boolq": ["passage", "question"]}[task]
    for obj in [example, example.get("text", {})]:
        if isinstance(obj, dict) and all(isinstance(obj.get(key), str) for key in keys):
            return {key: obj[key] for key in keys}
    # Experiment 1 caches the original input as one JSON object inside the prompt.
    for line in example.get("prompt", "").splitlines():
        if line.lstrip().startswith("{"):
            try:
                obj, _ = json.JSONDecoder().raw_decode(line.lstrip())
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and all(isinstance(obj.get(key), str) for key in keys):
                return {key: obj[key] for key in keys}
    raise ValueError(f"Cannot extract {task} input text for example {example.get('id')}. "
                     "Use the original Experiment 1 examples.json with its cached prompt texts.")


def features(example, task, args):
    text = extract_text(example, task)
    row = {"task": task, "id": str(example["id"])}
    if task == "snli":
        premise, hypothesis = tokens(text["premise"]), tokens(text["hypothesis"])
        row.update(hypothesis_tokens=len(hypothesis), negation=negated(hypothesis),
                   lexical_overlap=overlap(premise, hypothesis), query_content_tokens=len(set(hypothesis)-STOPWORDS),
                   reference_content_tokens=len(set(premise)-STOPWORDS))
        assignments = {"negation": "present" if row["negation"] else "absent",
                       "length": "short" if len(hypothesis) <= args.snli_short else "long",
                       "overlap": "high" if row["lexical_overlap"] >= args.overlap_threshold else "low"}
    elif task == "sst2":
        words = tokens(text["review"])
        row.update(review_tokens=len(words), negation=negated(words), contrast=bool(set(words) & CONTRAST))
        assignments = {"negation": "present" if row["negation"] else "absent",
                       "contrast": "present" if row["contrast"] else "absent",
                       "length": "short" if len(words) <= args.sst2_short else "long"}
    else:
        passage, question = tokens(text["passage"]), tokens(text["question"])
        first = question[0] if question else ""
        question_form = ("be" if first in {"is", "are", "was", "were", "am"} else
                         "do" if first in {"do", "does", "did"} else
                         "modal" if first in {"can", "could", "will", "would", "should", "may", "might", "must", "shall"} else
                         "have" if first in {"have", "has", "had"} else "other")
        row.update(passage_tokens=len(passage), question_tokens=len(question), question_form=question_form,
                   lexical_overlap=overlap(passage, question), query_content_tokens=len(set(question)-STOPWORDS),
                   reference_content_tokens=len(set(passage)-STOPWORDS))
        assignments = {"question_form": question_form,
                       "length": "short" if len(passage) <= args.boolq_short else "long",
                       "overlap": "high" if row["lexical_overlap"] >= args.overlap_threshold else "low"}
    row.update({"group_"+family: label for family, label in assignments.items()})
    return row, {"overall": "all", **assignments}


def group_definitions(task, args):
    result = [{"family": "overall", "label": "all", "definition": "All cached labeled examples"}]
    if task in ["snli", "sst2"]:
        field = "hypothesis" if task == "snli" else "review"
        result.extend({"family": "negation", "label": name, "definition": f"{field}: lexical negation cue {name}"}
                      for name in ["present", "absent"])
    if task == "sst2":
        result.extend({"family": "contrast", "label": name, "definition": f"Review: lexical contrast cue {name}"}
                      for name in ["present", "absent"])
    if task == "boolq":
        result.extend({"family": "question_form", "label": name, "definition": "First question token auxiliary class: "+name}
                      for name in ["be", "do", "modal", "have", "other"])
    threshold = {"snli": args.snli_short, "sst2": args.sst2_short, "boolq": args.boolq_short}[task]
    field = {"snli": "hypothesis", "sst2": "review", "boolq": "passage"}[task]
    result.extend({"family": "length", "label": name,
                   "definition": f"{field}: {'<=' if name == 'short' else '>'} {threshold} regex word tokens"}
                  for name in ["short", "long"])
    if task in ["snli", "boolq"]:
        result.extend({"family": "overlap", "label": name,
                       "definition": f"Unique query content-token overlap {'>=' if name == 'high' else '<'} {args.overlap_threshold:g}"}
                      for name in ["high", "low"])
    return result


def prepare_targets(args):
    targets, manifests, feature_rows, accuracy_rows = [], [], [], []
    for task in args.tasks:
        folder = Path(args.cache)/task
        examples = c.read_json(folder/"examples.json")
        rows_and_memberships = [features(example, task, args) for example in examples]
        feature_rows.extend(row for row, _ in rows_and_memberships)
        memberships = [membership for _, membership in rows_and_memberships]
        groups = []
        for definition in group_definitions(task, args):
            family, label = definition["family"], definition["label"]
            mask = np.array([member.get(family) == label for member in memberships])
            count = int(mask.sum())
            eligible = count > 0 and (family == "overall" or count >= args.min_examples)
            info = {"task": task, "subgroup": family+"_"+label, "family": family, "group_label": label,
                    "definition": definition["definition"], "group_examples": count,
                    "dataset_examples": len(examples), "group_fraction": count/len(examples),
                    "eligible": eligible, "skip_reason": "" if eligible else "empty" if count == 0 else "below_min_examples"}
            manifests.append(info)
            groups.append((mask, info))
        # The shared loader validates IDs, labels, completion, booleans and cache hashes.
        full_pairs = list(c.paired_frequencies(Path(args.cache), task, args.models))
        predictions = {}
        for model in args.models:
            records = {str(r["id"]): r for r in [json.loads(line) for line in (folder/(c.slug(model)+".jsonl")).read_text(encoding="utf-8").splitlines() if line.strip()]}
            predictions[model] = np.array([records[str(ex["id"])]["correct"] for ex in examples])
        for _, full_meta in full_pairs:
            ma, mb = full_meta["model_a"], full_meta["model_b"]
            for mask, info in groups:
                ca, cb = predictions[ma][mask], predictions[mb][mask]
                counts = np.array([(ca & ~cb).sum(), (~ca & cb).sum(), (ca == cb).sum()])
                f = counts/len(ca) if len(ca) else np.full(3, np.nan)
                gap = f[0]-f[1]
                metadata = {**full_meta, **info, "benchmark_examples": len(ca),
                            "accuracy_a": float(ca.mean()) if len(ca) else math.nan,
                            "accuracy_b": float(cb.mean()) if len(cb) else math.nan,
                            "a_only_count": int(counts[0]), "b_only_count": int(counts[1]), "tie_count": int(counts[2]),
                            "f_a_only": f[0], "f_b_only": f[1], "f_tie": f[2], "true_gap": gap,
                            "overall_gap": full_meta["true_gap"],
                            "empirical_ranking_reversal": bool(np.isfinite(gap) and gap*full_meta["true_gap"] < 0),
                            "group_target_id": c.digest([task, ma, mb, info["subgroup"], info["definition"],
                                                         np.flatnonzero(mask).tolist(), full_meta["cache_pair_sha256"]])[:20]}
                accuracy_rows.append(metadata)
                if info["eligible"]:
                    targets.append((f, metadata))
    return targets, manifests, feature_rows, accuracy_rows


def allocate_reports(targets, budget, mode):
    """Allocate within each task/model-pair/partition family.

    Equal mode spends budget per group. Proportional mode uses Hamilton allocation
    of one family budget over eligible groups; excluded groups get no reports.
    """
    if mode == "equal":
        counts = np.full(len(targets), budget, dtype=int)
    else:
        counts = np.zeros(len(targets), dtype=int)
        families = {}
        for i, (_, meta) in enumerate(targets):
            families.setdefault((meta["task"], meta["model_a"], meta["model_b"], meta["family"]), []).append(i)
        for indices in families.values():
            weights = np.array([targets[i][1]["group_examples"] for i in indices], dtype=float)
            raw = budget*weights/weights.sum()
            allocated = np.floor(raw).astype(int)
            remaining = budget-int(allocated.sum())
            order = np.argsort(-(raw-allocated), kind="stable")
            allocated[order[:remaining]] += 1
            counts[indices] = allocated
    totals = {}
    for i, (_, meta) in enumerate(targets):
        key = (meta["task"], meta["model_a"], meta["model_b"], meta["family"])
        totals[key] = totals.get(key, 0)+int(counts[i])
    return counts, [totals[(meta["task"], meta["model_a"], meta["model_b"], meta["family"])] for _, meta in targets]


def simulate(args):
    out = Path(args.out)
    if any(out.glob("*")) and not args.overwrite:
        raise ValueError(f"{out} is not empty. Choose a new --out or use --overwrite.")
    targets, manifests, feature_rows, accuracy_rows = prepare_targets(args)
    if not targets:
        raise ValueError("No eligible groups.")
    K = len(targets)
    # Allocated across ALL selected task/pair/group targets and BOTH methods.
    component_delta = args.delta/(2*K) if args.coverage_scope == "simultaneous" else args.delta
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k != "function"}
    config.update(version="experiment4-v1", script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  core_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),
                  python=c.platform.python_version(), packages={k: c.importlib.metadata.version(k) for k in ["numpy", "pandas", "scipy", "matplotlib"]},
                  target="Uniform empirical distribution conditional on each retained, input-defined subgroup",
                  groups="Lexical/length proxies, predefined without using model correctness; overlapping across families",
                  sampling="Separate conditional IID report collections; fixed honest membership in each stratum",
                  privacy_scope="Category-level LDP; task and subgroup routing identities are assumed public, not privatized",
                  contamination_scope="Assumed bound applies within EVERY subgroup collection, not merely globally",
                  coverage_targets=K, component_delta=component_delta,
                  simultaneous_scope="All selected task/pair/group targets and both methods, at ONE fixed channel, epsilon, budget, budget mode, contamination and attack",
                  attacks="Fixed weaker-model and random reporting; no adaptive attacks in Experiment 4")
    c.write_json(out/"configuration.json", config)
    c.atomic_csv(manifests, out/"group_manifest.csv")
    c.atomic_csv(feature_rows, out/"example_features.csv")
    c.atomic_csv(accuracy_rows, out/"subgroup_accuracy.csv")
    c.atomic_csv([row for row in accuracy_rows if row["family"] == "overall"], out/"model_accuracy.csv")
    allocation_rows = []
    for mode, budget in itertools.product(args.budget_modes, args.report_counts):
        counts, family_totals = allocate_reports(targets, budget, mode)
        allocation_rows.extend({"group_target_id": meta["group_target_id"], "task": meta["task"], "model_a": meta["model_a"],
                                "model_b": meta["model_b"], "subgroup": meta["subgroup"], "family": meta["family"],
                                "budget_mode": mode, "budget": budget, "N_group": int(counts[i]),
                                "family_total_reports": family_totals[i], "sampled": bool(counts[i] > 0)}
                               for i, (_, meta) in enumerate(targets))
    c.atomic_csv(allocation_rows, out/"budget_allocations.csv")
    trial_stream = gzip.open(out/"trials.csv.gz", "wt", encoding="utf-8", newline="")
    summary_stream = (out/"summary.csv").open("w", encoding="utf-8", newline="")
    joint_rows = []
    header = True
    grid = list(itertools.product(args.channels, args.epsilons, args.report_counts, args.budget_modes,
                                  args.contamination, args.attacks))
    try:
        try:
            from tqdm import tqdm
            grid = tqdm(grid, desc=f"Evaluate {K} subgroup targets")
        except ImportError:
            pass
        for channel, eps, budget, mode, nominal_b, attack in grid:
            if nominal_b == 0 and attack != args.attacks[0]:
                continue
            p, q = c.channel_params(channel, eps)
            bbar = nominal_b if args.bbar is None else args.bbar
            N_groups, family_totals = allocate_reports(targets, budget, mode)
            joint_gap = np.ones(args.repetitions, dtype=bool)
            joint_region_and_gap = np.ones(args.repetitions, dtype=bool)
            any_error = np.zeros(args.repetitions, dtype=bool)
            targets_evaluated, valid_joint = 0, True
            for i, (f, meta) in enumerate(targets):
                N = int(N_groups[i])
                if N == 0:
                    continue  # Zero-budget groups are explicitly listed in budget_allocations.csv.
                targets_evaluated += 1
                m = int((Decimal(N)*Decimal(str(nominal_b))).to_integral_value(rounding=ROUND_FLOOR))
                n = N-m
                valid = m <= int((Decimal(N)*Decimal(str(bbar))).to_integral_value(rounding=ROUND_FLOOR))
                valid_joint &= valid
                identity = {"group_target_id": meta["group_target_id"], "channel": channel, "epsilon": eps,
                            "N_group": N, "budget": budget, "budget_mode": mode, "nominal_b": nominal_b, "seed": args.seed}
                rng = np.random.default_rng(int(c.digest(identity)[:16], 16))
                honest = c.draw_honest_counts(rng, n, f, p, q, channel, args.repetitions)
                malicious = np.zeros_like(honest)
                if m:
                    if attack == "random":
                        arng = np.random.default_rng(int(c.digest([identity, "random_attack"])[:16], 16))
                        malicious = (arng.multinomial(m, [1/3]*3, size=args.repetitions) if channel == "GRR" else
                                     arng.binomial(m, .5, size=honest.shape))
                    else:
                        malicious[:, 0 if f[0] <= f[1] else 1] = m
                observed = honest+malicious
                s = observed/N
                clo, chi, t = c.theorem1_bounds(s, N, p, q, bbar, component_delta)
                l1, u1, feasible1 = c.query_extrema(clo, chi)
                l3, u3, feasible3, radii = c.theorem3_bounds(s, N, p, q, bbar, component_delta, channel)
                distribution_covered = (f >= clo).all(axis=1) & (f <= chi).all(axis=1)
                setting_id = c.digest([identity, bbar, attack, component_delta])[:20]
                base = {**meta, "setting_id": setting_id, "channel": channel, "epsilon": eps,
                        "budget": budget, "budget_mode": mode, "N": N, "N_group": N, "family_total_reports": family_totals[i],
                        "all_selected_targets_total_reports": int(N_groups.sum()),
                        "n": n, "malicious_count": m, "nominal_b": nominal_b, "actual_b": m/N, "bbar": bbar,
                        "assumption_satisfied": valid, "attack": attack if m else "none", "requested_attack": attack,
                        "delta": args.delta, "component_delta": component_delta, "coverage_scope": args.coverage_scope,
                        "coverage_targets": K, "repetitions": args.repetitions, "seed": args.seed,
                        "p": p, "q": q, "t_coordinate": t, **radii}
                trial = {"setting_id": setting_id, "trial": np.arange(args.repetitions),
                         **{key: base[key] for key in ["group_target_id", "task", "model_a", "model_b", "subgroup", "family",
                                                      "channel", "epsilon", "budget", "budget_mode", "N_group", "malicious_count",
                                                      "nominal_b", "actual_b", "bbar", "assumption_satisfied", "attack", "true_gap",
                                                      "overall_gap", "delta", "component_delta"]},
                         "theorem1_distribution_covered": distribution_covered}
                for coordinate in range(3):
                    trial[f"honest_support_{coordinate+1}"] = honest[:, coordinate]
                    trial[f"malicious_support_{coordinate+1}"] = malicious[:, coordinate]
                    trial[f"observed_support_{coordinate+1}"] = observed[:, coordinate]
                    trial[f"theorem1_coordinate_lower_{coordinate+1}"] = clo[:, coordinate]
                    trial[f"theorem1_coordinate_upper_{coordinate+1}"] = chi[:, coordinate]
                rows = []
                truth = int(np.sign(meta["true_gap"]))
                for method, lo, hi, feasible in [("theorem1", l1, u1, feasible1), ("theorem3", l3, u3, feasible3)]:
                    d = c.decision(lo, hi, feasible)
                    covered = feasible & (lo <= meta["true_gap"]) & (meta["true_gap"] <= hi)
                    wrong, correct = (d != 0) & (d != truth), (d != 0) & (d == truth)
                    joint_gap &= covered
                    joint_region_and_gap &= distribution_covered if method == "theorem1" else covered
                    any_error |= wrong
                    reversal = (d != 0) & (d == -int(np.sign(meta["overall_gap"]))) if meta["overall_gap"] != 0 else np.zeros(len(d), dtype=bool)
                    row = {**base, "method": method, "feasible_trials": int(feasible.sum()),
                           **c.rate_stats(covered, "gap_coverage"), **c.rate_stats(wrong, "error"),
                           **c.rate_stats(correct, "correct_certification"), **c.rate_stats(d != 0, "certification"),
                           **c.rate_stats(d == 0, "abstention"), **c.rate_stats(~feasible, "empty_interval"),
                           **c.rate_stats(reversal & correct, "correct_certified_reversal"),
                           **c.quantiles(lo[feasible], "gap_lower"), **c.quantiles(hi[feasible], "gap_upper"),
                           **c.quantiles((hi-lo)[feasible], "gap_width")}
                    row.update(c.rate_stats(distribution_covered, "distribution_coverage") if method == "theorem1" else
                               {"distribution_coverage_"+key: math.nan for key in ["count", "rate", "ci_low", "ci_high"]})
                    rows.append(row)
                    trial.update({method+"_lower": lo, method+"_upper": hi, method+"_feasible": feasible,
                                  method+"_decision": d, method+"_gap_covered": covered, method+"_error": wrong,
                                  method+"_correct_certification": correct})
                pd.DataFrame(trial).to_csv(trial_stream, header=header, index=False)
                pd.DataFrame(rows).to_csv(summary_stream, header=header, index=False)
                summary_stream.flush()
                header = False
            joint_rows.append({"channel": channel, "epsilon": eps, "budget": budget, "budget_mode": mode,
                               "nominal_b": nominal_b, "bbar": bbar, "attack": attack if nominal_b else "none",
                               "delta": args.delta, "component_delta": component_delta,
                               "coverage_scope": args.coverage_scope, "coverage_targets": K,
                               "all_selected_targets_total_reports": int(N_groups.sum()),
                               "targets_evaluated": targets_evaluated, "repetitions": args.repetitions,
                               "joint_guarantee_available": args.coverage_scope == "simultaneous" and valid_joint,
                               "assumption_satisfied": valid_joint,
                               **c.rate_stats(joint_gap, "joint_gap_coverage"),
                               **c.rate_stats(joint_region_and_gap, "joint_region_and_gap_coverage"),
                               **c.rate_stats(any_error, "any_error")})
            c.atomic_csv(joint_rows, out/"simultaneous_coverage.csv")
            trial_stream.flush()
    finally:
        trial_stream.close()
        summary_stream.close()
    print(f"Saved {out/'summary.csv'}, {out/'trials.csv.gz'}, subgroup details and simultaneous_coverage.csv.")
    print(f"{K} selected task/pair/group targets; scope={args.coverage_scope}; per-method/target delta={component_delta:g}")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(args.csv)
    required = {"setting_id", "task", "model_a", "model_b", "method", "subgroup", "family", "group_examples", "true_gap",
                "overall_gap", "channel", "epsilon", "budget", "budget_mode", "N_group", "nominal_b", "bbar", "attack",
                "requested_attack", "delta", "component_delta", "coverage_scope", "assumption_satisfied",
                "gap_lower_median", "gap_upper_median", "gap_width_median", "correct_certification_rate",
                "correct_certification_ci_low", "correct_certification_ci_high", "gap_coverage_rate", "error_rate"}
    required.update(metric+suffix for metric in ["gap_coverage", "error"] for suffix in ["_ci_low", "_ci_high"])
    if required-set(df):
        raise ValueError("Not an Experiment 4 summary CSV; missing "+", ".join(sorted(required-set(df))))
    if df.duplicated(["setting_id", "method"]).any():
        raise ValueError("Duplicate setting/method rows.")
    if args.joint_csv and args.tasks:
        raise ValueError("Joint coverage concerns all selected tasks/pairs/groups in the run; omit --tasks when supplying --joint-csv.")
    for option, col in [("tasks", "task"), ("channels", "channel"), ("epsilons", "epsilon"), ("contamination", "nominal_b"), ("budget_modes", "budget_mode")]:
        values = getattr(args, option)
        if values:
            if col in ["epsilon", "nominal_b"]:
                df = df[np.logical_or.reduce([np.isclose(df[col], value, atol=1e-12, rtol=0) for value in values])]
            else:
                df = df[df[col].isin(values)]
    df = df[df.requested_attack.isin(args.attacks)]
    if df.empty:
        raise ValueError("No rows match the requested plot filters.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figure_index, budget_rows = [], []
    colors = {"theorem1": "#2458a6", "theorem3": "#d46b19"}

    def save(fig, stem, info):
        fig.savefig(out/(stem+".pdf"), bbox_inches="tight")
        fig.savefig(out/(stem+".png"), dpi=200, bbox_inches="tight")
        figure_index.append({"figure_stem": stem, **info})
        plt.close(fig)

    groups = ["task", "model_a", "model_b", "channel", "epsilon", "budget_mode", "nominal_b", "bbar", "requested_attack", "coverage_scope", "component_delta"]
    for key, sub in df.groupby(groups, dropna=False):
        task, ma, mb, channel, eps, mode, nominal_b, bbar, attack, scope, component_delta = key
        valid = sub.assumption_satisfied.astype(str).str.lower().isin(["true", "1"]).all()
        title = f"{task.upper()} | {ma} vs {mb}\n{channel}, epsilon={eps:g}, b={nominal_b:g}, bound={bbar:g}, {mode} budgets | {attack}"
        info = dict(zip(groups, key))
        stem = f"{task}_{channel}_eps{eps:g}_{mode}_b{nominal_b:g}_{c.digest(list(key))[:8]}"
        budget = args.budget or int(sub.budget.max())
        selected = sub[sub.budget == budget]
        if selected.empty:
            raise ValueError(f"Budget {budget} is not present for {task}/{channel}/epsilon={eps}.")
        group_order = selected.drop_duplicates("subgroup").sort_values(["family", "subgroup"]).subgroup.tolist()
        group_order = ["overall_all"]+[group for group in group_order if group != "overall_all"]
        fig, ax = plt.subplots(figsize=(9, max(4, .42*len(group_order)+1.5)))
        positions = np.arange(len(group_order))
        for method, offset in [("theorem1", -.13), ("theorem3", .13)]:
            method_rows = selected[selected.method == method].set_index("subgroup").reindex(group_order)
            lo, hi = method_rows.gap_lower_median.to_numpy(), method_rows.gap_upper_median.to_numpy()
            midpoint = (lo+hi)/2
            ax.errorbar(midpoint, positions+offset, xerr=np.maximum(0, np.vstack([midpoint-lo, hi-midpoint])), fmt="o", ms=3,
                        capsize=2, color=colors[method], label=method)
        unique = selected.drop_duplicates("subgroup").set_index("subgroup").reindex(group_order)
        ax.scatter(unique.true_gap, positions, marker="x", color="black", s=25, label="True empirical subgroup gap", zorder=5)
        ax.axvline(0, color="black", lw=.8)
        ax.axvline(sub.overall_gap.iloc[0], color="gray", lw=.8, ls="--", label="Overall empirical gap")
        ax.set_yticks(positions, [f"{group} (examples={int(unique.loc[group, 'group_examples'])}, reports={int(unique.loc[group, 'N_group'])})" for group in group_order], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Accuracy gap A - B")
        ax.grid(axis="x", alpha=.2)
        ax.legend(fontsize=7, loc="best")
        fig.suptitle(title+f"\nBudget={budget:,}; failure scope={scope}", fontsize=10)
        fig.text(.01, .01, "Segments summarize median endpoints among nonempty trials; NOT single confidence intervals. Missing segment = no nonempty trials.", fontsize=7)
        fig.tight_layout(rect=(0, .06, 1, .89))
        save(fig, stem+"_subgroup_gaps", {**info, "budget": budget, "figure_type": "subgroup_gaps"})

        fig, axes = plt.subplots(1, 2, figsize=(12, max(4, .35*len(group_order)+1.6)), squeeze=False)
        for ax, method in zip(axes[0], METHODS):
            heat = sub[sub.method == method].pivot(index="subgroup", columns="budget", values="correct_certification_rate").reindex(group_order).sort_index(axis=1)
            im = ax.imshow(np.ma.masked_invalid(heat.to_numpy()), aspect="auto", origin="upper", vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(heat.columns)), [f"{int(value):,}" for value in heat.columns], rotation=45, ha="right")
            ax.set_yticks(range(len(heat.index)), heat.index, fontsize=8)
            ax.set_xlabel("Per-group N" if mode == "equal" else "Total budget per partition family")
            ax.set_title(method)
            fig.colorbar(im, ax=ax, fraction=.046, label="Correct certification rate")
        fig.suptitle(title, fontsize=10)
        fig.text(.01, .01, "Groups overlap across families. Heatmap cells use the recorded subgroup report allocation; no averaging of budgets or groups.", fontsize=7)
        fig.tight_layout(rect=(0, .07, 1, .88))
        save(fig, stem+"_certification_heatmap", {**info, "figure_type": "certification_heatmap"})

        fig, axes = plt.subplots(1, 2, figsize=(12, max(4, .35*len(group_order)+1.6)), squeeze=False)
        x = np.arange(len(group_order))
        for method, offset in [("theorem1", -.14), ("theorem3", .14)]:
            rows = selected[selected.method == method].set_index("subgroup").reindex(group_order)
            for ax, metric in zip(axes[0], ["gap_coverage", "error"]):
                values = rows[metric+"_rate"].to_numpy()
                intervals = np.maximum(0, np.vstack([values-rows[metric+"_ci_low"].to_numpy(), rows[metric+"_ci_high"].to_numpy()-values]))
                ax.errorbar(x+offset, values, yerr=intervals, fmt="o", ms=4, capsize=2, color=colors[method], label=method)
        for ax, label, threshold in [(axes[0, 0], "Gap coverage rate", 1-component_delta), (axes[0, 1], "Incorrect certification rate", component_delta)]:
            ax.axhline(threshold, color="black", ls="--", lw=.8)
            ax.set_xticks(x, group_order, rotation=60, ha="right", fontsize=7)
            ax.set_ylabel(label)
            ax.set_ylim(-.02, 1.02)
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(title+f"\nBudget={budget:,}; component delta={component_delta:.3g}", fontsize=10)
        fig.text(.01, .01, ("Bound satisfied." if valid else "Some bounds violated: theorem guarantees do not apply there.")+" Error bars: pointwise 95% Monte Carlo intervals.", fontsize=7)
        fig.tight_layout(rect=(0, .07, 1, .88))
        save(fig, stem+"_coverage_error", {**info, "budget": budget, "figure_type": "coverage_error"})
        for (subgroup, method), curve in sub.groupby(["subgroup", "method"]):
            curve = curve.sort_values("budget")
            hits = curve[curve.correct_certification_ci_low >= args.target_rate]
            budget_rows.append({**info, "subgroup": subgroup, "method": method, "target_rate": args.target_rate,
                                "first_tested_budget": int(hits.budget.iloc[0]) if len(hits) else math.nan,
                                "N_group_at_first_tested_budget": int(hits.N_group.iloc[0]) if len(hits) else math.nan,
                                "status": "reached" if len(hits) else "not reached",
                                "criterion": "pointwise 95% Monte Carlo lower bound >= target"})
    c.atomic_csv(budget_rows, out/"reports_required.csv")
    c.atomic_csv(figure_index, out/"figure_index.csv")
    if args.joint_csv:
        plot_joint(args, out)
    print(f"Saved figures, reports_required.csv and figure_index.csv in {out}. Plotting used saved CSVs only.")


def plot_joint(args, out):
    import matplotlib.pyplot as plt
    df = pd.read_csv(args.joint_csv)
    required = {"channel", "epsilon", "budget", "budget_mode", "nominal_b", "bbar", "attack", "delta",
                "coverage_scope", "joint_guarantee_available", "joint_region_and_gap_coverage_rate",
                "joint_region_and_gap_coverage_ci_low", "joint_region_and_gap_coverage_ci_high",
                "any_error_rate", "any_error_ci_low", "any_error_ci_high"}
    if required-set(df):
        raise ValueError("Missing columns in simultaneous_coverage.csv: "+", ".join(sorted(required-set(df))))
    if args.tasks:
        raise ValueError("--joint-csv records the whole run's task/pair/group family; omit --tasks to plot that joint guarantee.")
    for option, col in [("channels", "channel"), ("epsilons", "epsilon"), ("contamination", "nominal_b"), ("budget_modes", "budget_mode")]:
        values = getattr(args, option)
        if values:
            if col in ["epsilon", "nominal_b"]:
                df = df[np.logical_or.reduce([np.isclose(df[col], value, atol=1e-12, rtol=0) for value in values])]
            else:
                df = df[df[col].isin(values)]
    df = df[df.attack.isin(args.attacks+["none"])]
    keys = ["channel", "epsilon", "budget_mode", "nominal_b", "bbar", "attack", "coverage_scope", "delta"]
    for key, sub in df.groupby(keys):
        channel, eps, mode, nominal_b, bbar, attack, scope, delta = key
        sub = sub.sort_values("budget")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        for ax, metric, threshold, title in [(axes[0], "joint_region_and_gap_coverage", 1-delta, "All targets/methods cover"),
                                            (axes[1], "any_error", delta, "Any incorrect certificate")]:
            ax.plot(sub.budget, sub[metric+"_rate"], "o-", ms=4)
            ax.fill_between(sub.budget.to_numpy(), sub[metric+"_ci_low"].to_numpy(), sub[metric+"_ci_high"].to_numpy(), alpha=.15)
            if scope == "simultaneous":
                ax.axhline(threshold, color="black", ls="--", lw=.8)
            ax.set_xscale("log")
            ax.set_ylim(-.02, 1.02)
            ax.set_xlabel("Per-group budget" if mode == "equal" else "Budget per partition family")
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=.2)
        available = sub.joint_guarantee_available.astype(str).str.lower().isin(["true", "1"]).all()
        fig.suptitle(f"Whole selected task/pair/group family | {channel}, epsilon={eps:g}\n{mode}, b={nominal_b:g}, bound={bbar:g}, {attack}; scope={scope}", fontsize=10)
        fig.text(.01, .01, ("Joint guarantee applies." if available else "No joint guarantee claimed for these settings.")+" Shading: pointwise 95% Monte Carlo intervals.", fontsize=7)
        fig.tight_layout(rect=(0, .08, 1, .83))
        stem = f"joint_{channel}_eps{eps:g}_{mode}_b{nominal_b:g}_{c.digest(list(key))[:8]}"
        fig.savefig(out/(stem+".pdf"), bbox_inches="tight")
        fig.savefig(out/(stem+".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--cache", required=True)
    s.add_argument("--out", default="experiment4_results")
    s.add_argument("--tasks", nargs="+", choices=["snli", "sst2", "boolq"], default=["snli"])
    s.add_argument("--models", nargs="+", default=c.MODELS)
    s.add_argument("--channels", nargs="+", choices=["GRR", "OUE"], default=["GRR", "OUE"])
    s.add_argument("--epsilons", nargs="+", type=float, default=[.1, .5, 1, 2, 5])
    s.add_argument("--report-counts", nargs="+", type=int, default=[1000, 10000, 100000, 1000000])
    s.add_argument("--budget-modes", nargs="+", choices=["equal", "proportional"], default=["equal", "proportional"])
    s.add_argument("--contamination", nargs="+", type=float, default=[0, .001, .01])
    s.add_argument("--attacks", nargs="+", choices=["favor_weaker", "random"], default=["favor_weaker", "random"])
    s.add_argument("--bbar", type=float, default=None)
    s.add_argument("--delta", type=float, default=.05, help="Family failure level under simultaneous scope")
    s.add_argument("--coverage-scope", choices=["simultaneous", "individual"], default="simultaneous")
    s.add_argument("--min-examples", type=int, default=30)
    s.add_argument("--snli-short", type=int, default=10)
    s.add_argument("--sst2-short", type=int, default=20)
    s.add_argument("--boolq-short", type=int, default=150)
    s.add_argument("--overlap-threshold", type=float, default=.5)
    s.add_argument("--repetitions", type=int, default=1000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(function=simulate)
    t = sub.add_parser("plot")
    t.add_argument("--csv", required=True)
    t.add_argument("--joint-csv", default=None, help="Optional simultaneous_coverage.csv to plot family-level reliability")
    t.add_argument("--out", default="experiment4_figures")
    t.add_argument("--tasks", nargs="+", choices=["snli", "sst2", "boolq"])
    t.add_argument("--channels", nargs="+", choices=["GRR", "OUE"])
    t.add_argument("--epsilons", nargs="+", type=float)
    t.add_argument("--contamination", nargs="+", type=float)
    t.add_argument("--budget-modes", nargs="+", choices=["equal", "proportional"])
    t.add_argument("--attacks", nargs="+", choices=["favor_weaker", "random"], default=["favor_weaker"])
    t.add_argument("--budget", type=int, default=None, help="Snapshot budget; default largest tested budget in each panel")
    t.add_argument("--target-rate", type=float, default=.9)
    t.set_defaults(function=plot)
    return p


def main():
    p = parser()
    args = p.parse_args()
    for key in ["tasks", "models", "channels", "epsilons", "report_counts", "budget_modes", "contamination", "attacks"]:
        values = getattr(args, key, None)
        if values is not None and len(values) != len(set(values)):
            p.error("Duplicate values in --"+key.replace("_", "-"))
    if args.command == "simulate":
        if len(args.models) < 2 or args.repetitions < 1 or args.seed < 0 or args.min_examples < 1:
            p.error("Require >=2 models, positive repetitions/min-examples, and nonnegative seed.")
        if not math.isfinite(args.delta) or not 0 < args.delta < 1:
            p.error("delta must be in (0,1).")
        if any(N < 1 for N in args.report_counts) or min(args.snli_short, args.sst2_short, args.boolq_short) < 1:
            p.error("Report budgets and token thresholds must be positive.")
        if not math.isfinite(args.overlap_threshold) or not 0 < args.overlap_threshold <= 1:
            p.error("Overlap threshold must be in (0,1].")
        if any(not math.isfinite(b) or not 0 <= b < 1 for b in args.contamination):
            p.error("Contamination must be in [0,1).")
        if args.bbar is not None and (not math.isfinite(args.bbar) or not 0 <= args.bbar < 1):
            p.error("bbar must be in [0,1).")
        for eps in args.epsilons:
            c.channel_params("GRR", eps)
    elif not math.isfinite(args.target_rate) or not 0 < args.target_rate < 1:
        p.error("target-rate must be in (0,1).")
    try:
        args.function(args)
    except (ValueError, FileNotFoundError) as exc:
        p.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
