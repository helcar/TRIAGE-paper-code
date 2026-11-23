"""Report practical evaluation tables from frozen, aligned test scores.

All metrics are computed on EXISTING test predictions. No test fitting or
parameter search is performed. Dataset score CSV must be produced by
code.evaluate_ffpp_baselines and correspond to the supplied manifest.

Usage:
 python -m code.report_practical_results \
   --dataset FFPP D:/TRIAGE/ffpp_c23_pilot_manifest.csv results/ffpp_c23_selected_baselines_scores.csv \
   --output results/practical_results.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from .evaluation import evaluate

METHODS = (
    "Prediction Confidence", "Feature Centroid", "Mahalanobis",
    "kNN", "Prototype Attribution Divergence", "TRIAGE"
)


def load_dataset(manifest_file, scores_file):
    with open(manifest_file, newline="", encoding="utf-8-sig") as f:
        manifest = [r for r in csv.DictReader(f) if r["split"] == "test"]
    with open(scores_file, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not manifest or len(rows) != len(manifest):
        raise ValueError("Test manifest and score CSV have unequal/zero rows")
    y = np.empty(len(rows), dtype=int)
    error = np.empty(len(rows), dtype=int)
    groups = []
    score = {k: np.empty(len(rows), dtype=float) for k in METHODS}
    for i, (a, b) in enumerate(zip(manifest, rows)):
        if a["group_id"] != b["group_id"] or a["video_id"] != b["video_id"]:
            raise ValueError(f"Row alignment mismatch at index {i}")
        if a["label"] not in ("0", "1"):
            raise ValueError(f"Nonbinary label at {i}")
        y[i], error[i] = int(a["label"]), int(b["error"])
        if error[i] not in (0, 1):
            raise ValueError(f"Nonbinary error flag at {i}")
        groups.append(a["group_id"])
        for k in METHODS:
            score[k][i] = float(b[k])
    for name, values in score.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite scores in {name}")
    confidence_risk = score["Prediction Confidence"]
    if np.any(confidence_risk < -1e-7) or np.any(confidence_risk > .5+1e-7):
        raise ValueError("Confidence risk outside [0,0.5]")
    return y, error, np.asarray(groups), score


def subset_metrics(error, scores, indices):
    return {name: evaluate(error[indices], data[indices])
            for name, data in scores.items()}


def group_bootstrap_diff(error, score_a, score_b, groups, repeats, seed):
    """Cluster-paired bootstrap over source group IDs, difference A minus B.

    AUPRC intervals depend on resampled prevalence. Skip replicates without
    both error and correct classes. These are descriptive bootstrap CIs.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    u, inv = np.unique(groups, return_inverse=True)
    group_rows = [np.flatnonzero(inv == g) for g in range(len(u))]
    rng = np.random.default_rng(seed)
    auc, ap = [], []
    for _ in range(repeats):
        sampled = rng.integers(0, len(u), size=len(u))
        idx = np.concatenate([group_rows[g] for g in sampled])
        e = error[idx]
        if e.min() == e.max():
            continue
        auc.append(roc_auc_score(e, score_a[idx]) -
                   roc_auc_score(e, score_b[idx]))
        ap.append(average_precision_score(e, score_a[idx]) -
                  average_precision_score(e, score_b[idx]))
    def pack(diffs, observed):
        return {"difference": float(observed), "ci95":
                np.quantile(diffs, [.025, .975]).tolist() if len(diffs) >= 30 else None,
                "valid_replicates": len(diffs)}
    if not len(error) or np.unique(error).size != 2:
        return None
    return {
        "AUROC": pack(auc, roc_auc_score(error, score_a) - roc_auc_score(error, score_b)),
        "AUPRC": pack(ap, average_precision_score(error, score_a) -
                      average_precision_score(error, score_b))
    }


def report_dataset(name, manifest, scores_file, repeats, seed, budget):
    y, errors, groups, scores = load_dataset(manifest, scores_file)
    n = len(errors)
    top_n = max(1, int(np.ceil(budget*n)))
    report = {
        "n": n, "errors": int(errors.sum()), "source_groups": len(set(groups)),
        "class_counts": {"real": int((y == 0).sum()), "fake": int((y == 1).sum())},
        "overall": subset_metrics(errors, scores, np.ones(n, dtype=bool)),
        "by_true_class": {},
        "high_confidence_recovery": {"review_budget": budget, "review_count": top_n,
                                     "error_subgroups": {}, "methods": {}},
        "paired_group_bootstrap": {}
    }
    # Among real samples, error means false positive; among fake samples,
    # error means false negative. Correct predictions form negatives.
    for label, key in ((0, "FP_detection_among_real"), (1, "FN_detection_among_fake")):
        mask = y == label
        report["by_true_class"][key] = {
            "n": int(mask.sum()), "errors": int(errors[mask].sum()),
            "scores": subset_metrics(errors, scores, mask)
        }
    # High-confidence errors must be recovered from a single global review
    # queue, not a separate top-10%-within-subgroup queue.
    conf_risk = scores["Prediction Confidence"]
    conditions = {
        "all_errors": np.ones(n, dtype=bool),
        "errors_confidence_ge_090": conf_risk <= .10+1e-12,
        "errors_confidence_ge_095": conf_risk <= .05+1e-12,
    }
    for key, mask in conditions.items():
        report["high_confidence_recovery"]["error_subgroups"][key] = int((errors[mask] == 1).sum())
    for method, values in scores.items():
        # stable tie order; high scores go to review
        ranked = np.argsort(values, kind="stable")[-top_n:]
        selected = np.zeros(n, dtype=bool)
        selected[ranked] = True
        entry = {}
        for key, mask in conditions.items():
            denominator = int((errors[mask] == 1).sum())
            recovered = int((errors[mask & selected] == 1).sum())
            entry[key] = {"recovered": recovered, "eligible_errors": denominator,
                          "recall": recovered / denominator if denominator else None}
        report["high_confidence_recovery"]["methods"][method] = entry
    if repeats:
        for scope, mask in (("overall", np.ones(n, dtype=bool)),
                            ("low_confidence_lt_70pct", conf_risk > .30)):
            report["paired_group_bootstrap"][scope] = {
                "n": int(mask.sum()), "errors": int(errors[mask].sum()),
                "TRIAGE_minus_confidence": group_bootstrap_diff(
                    errors[mask], scores["TRIAGE"][mask],
                    scores["Prediction Confidence"][mask], groups[mask],
                    repeats, seed)
            }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", nargs=3, action="append", required=True,
                   metavar=("NAME", "MANIFEST", "BASELINE_SCORES_CSV"))
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap-repeats", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget", type=float, default=.10)
    args = p.parse_args()
    if not 0 < args.budget < 1 or args.bootstrap_repeats < 0:
        p.error("--budget must lie in (0,1), bootstrap repetitions nonnegative")
    results = {}
    for name, manifest, scores in args.dataset:
        if name in results:
            p.error(f"Duplicate dataset name {name}")
        results[name] = report_dataset(name, manifest, scores,
                                       args.bootstrap_repeats, args.seed,
                                       args.budget)
        r = results[name]
        print(f"\n{name}: n={r['n']}, errors={r['errors']}, groups={r['source_groups']}")
        print(f"{'Method':36} {'AURC':>8} {'Risk@90%':>10} {'FP AUROC':>10} {'FN AUROC':>10} {'Recall@10%':>11}")
        for method in METHODS:
            v = r["overall"][method]
            fp = r["by_true_class"]["FP_detection_among_real"]["scores"][method]
            fn = r["by_true_class"]["FN_detection_among_fake"]["scores"][method]
            recall = r["high_confidence_recovery"]["methods"][method]["all_errors"]["recall"]
            def fmt(x): return f"{x:.4f}" if x is not None else "N/A"
            print(f"{method:36} {fmt(v['aurc']):>8} {fmt(v['risk_at_coverage_0.9']):>10}"
                  f" {fmt(fp['auroc']):>10} {fmt(fn['auroc']):>10} {fmt(recall):>11}")
        for key, entry in r["paired_group_bootstrap"].items():
            if entry["TRIAGE_minus_confidence"] is not None:
                print(f"  Group bootstrap {key}: {entry['TRIAGE_minus_confidence']}")
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
