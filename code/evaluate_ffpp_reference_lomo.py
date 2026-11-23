"""FF++ reference-only leave-one-manipulation-out experiment.

The detector checkpoint is NEVER retrained. For each fake manipulation,
exclude its rows from TRIAGE fit/cal and validation. Score held-out fake
test frames and the official real test frames using the unchanged detector.
This tests reference construction under manipulation exclusion, NOT a
detector trained without that manipulation. Same real test frames reused
across holdouts; fold metrics must not be treated as independent.

Uses memory-mapped feature arrays and lazy index selection.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from .run_experiment import read, pred
from .triage import TRIAGE, Config
from .validate_triage import prototype_matrices
from .evaluation import (
    evaluate, error_labels, confidence_score, FeatureBaselines,
    fit_prototype_profiles, prototype_attribution_divergence
)


class LazySelect:
    def __init__(self, source, indices):
        self.source = source
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, selection):
        return self.source[self.indices[selection]]


def selected(data, indices):
    """Metadata compact; high-dimensional representations remain memmapped."""
    return {
        "labels": np.asarray(data["labels"][indices], dtype=int),
        "logits": np.asarray(data["logits"][indices], dtype=float),
        "features": LazySelect(data["features"], indices),
        "attributions": LazySelect(data["attributions"], indices),
    }


def mean_features(data, batch):
    result = []
    for i in range(0, len(data["labels"]), batch):
        result.append(np.asarray(data["features"][i:i+batch]).mean(axis=1))
    return np.vstack(result)


def score_fold(arrays, split_rows, holdout, cfg, batch):
    chosen = {}
    for split in ("fit", "cal", "val", "test"):
        rows = split_rows[split]
        if split == "test":
            indices = [i for i, r in enumerate(rows)
                       if r["label"] == "0" or r["method"] == holdout]
        else:
            indices = [i for i, r in enumerate(rows)
                       if r["label"] == "0" or r["method"] != holdout]
        if not indices:
            raise ValueError(f"Empty {split} for held-out {holdout}")
        chosen[split] = indices
    fit, cal, val, test = (selected(arrays[k], chosen[k])
                          for k in ("fit", "cal", "val", "test"))
    for split in ("fit", "cal", "val", "test"):
        rows = split_rows[split]
        selected_labels = [int(rows[i]["label"]) for i in chosen[split]]
        if not np.array_equal(chosen_labels := np.asarray(selected_labels),
                              selected(arrays[split], chosen[split])["labels"]):
            raise ValueError(f"Manifest labels do not match {split} arrays")
    # Verify selected source groups disjoint across all four partitions.
    sets = {k: {split_rows[k][i]["group_id"] for i in chosen[k]} for k in chosen}
    keys = list(sets)
    for a in range(len(keys)):
        for b in range(a+1, len(keys)):
            if sets[keys[a]] & sets[keys[b]]:
                raise ValueError(f"Group leakage between {keys[a]} and {keys[b]}")
    model = TRIAGE(cfg)
    model.fit_prototypes(
        fit["features"][i:i+batch]
        for i in range(0, len(fit["labels"]), batch))
    fp = prototype_matrices(model, fit, batch)
    cp = prototype_matrices(model, cal, batch)
    vp = prototype_matrices(model, val, batch)
    tp = prototype_matrices(model, test, batch)
    model.fit_topics(*fp)
    model.fit_references(model.encode(*cp), cal["labels"], pred(cal))
    ve = error_labels(val["labels"], pred(val))
    te = error_labels(test["labels"], pred(test))
    val_score = model.score(model.encode(*vp), pred(val))
    triage_score = model.score(model.encode(*tp), pred(test))
    scores = {"TRIAGE": triage_score,
              "Prediction Confidence": confidence_score(test["logits"])}
    feature = FeatureBaselines().fit(mean_features(cal, batch),
                                    cal["labels"], pred(cal))
    baseline = feature.score(mean_features(test, batch), pred(test))
    scores.update({
        "Feature Centroid": baseline["centroid"],
        "Mahalanobis": baseline["mahalanobis"],
        "kNN": baseline["knn"],
    })
    refs = fit_prototype_profiles(*cp, cal["labels"], pred(cal))
    scores["Prototype Attribution Divergence"] = prototype_attribution_divergence(
        *tp, pred(test), refs)
    source_val = evaluate(ve, val_score)
    target_scores = {key: evaluate(te, value) for key, value in scores.items()}
    return {
        "holdout_manipulation": holdout,
        "protocol": "unchanged detector, held-out method excluded from TRIAGE fit/cal/val; test contains held-out fake + official real",
        "n_fit": len(fit["labels"]), "n_cal": len(cal["labels"]),
        "n_val": len(val["labels"]), "n_test": len(te),
        "test_real": int((test["labels"] == 0).sum()),
        "test_fake": int((test["labels"] == 1).sum()),
        "n_errors": int(te.sum()), "source_validation": source_val,
        "target_test": target_scores,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    for split in ("fit", "cal", "val", "test"):
        p.add_argument("--" + split, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--holdout", nargs="+",
                   help="Exact method names; omit to use all fake methods")
    p.add_argument("--prototypes", type=int, default=64)
    p.add_argument("--topics", type=int, default=8)
    p.add_argument("--references", type=int, default=1)
    p.add_argument("--tau", type=float, default=10.)
    p.add_argument("--lambda-support", type=float, default=0.)
    p.add_argument("--lambda-oppose", type=float, default=1.)
    p.add_argument("--lambda-magnitude", type=float, default=.1)
    p.add_argument("--max-local-features", type=int, default=50000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    with open(args.manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    splits = {split: [r for r in rows if r["split"] == split]
              for split in ("fit", "cal", "val", "test")}
    available = sorted({r["method"] for r in rows if r["label"] == "1"})
    chosen = args.holdout or available
    if any(x not in available for x in chosen) or len(set(chosen)) != len(chosen):
        p.error(f"Unknown/duplicate holdout; available={available}")
    print(f"Fake manipulation methods: {available}", flush=True)
    arrays = {key: read(getattr(args, key)) for key in splits}
    for key in splits:
        if len(splits[key]) != len(arrays[key]["labels"]):
            raise ValueError(f"Split {key}: manifest/arrays length mismatch")
        # Exact manifest ordering is required; test against all labels.
        if not np.array_equal(
            np.fromiter((int(row["label"]) for row in splits[key]), dtype=int),
            np.asarray(arrays[key]["labels"])):
            raise ValueError(f"{key} labels misaligned")
    cfg = Config(prototypes=args.prototypes, topics_positive=args.topics,
                 topics_negative=args.topics, references_real=args.references,
                 references_fake=args.references, tau=args.tau,
                 lambda_support=args.lambda_support, lambda_oppose=args.lambda_oppose,
                 lambda_magnitude=args.lambda_magnitude,
                 max_local_features=args.max_local_features, seed=args.seed)
    summary = {}
    for method in chosen:
        print(f"\n=== Reference holdout: {method} ===", flush=True)
        result = score_fold(arrays, splits, method, cfg, args.batch)
        summary[method] = result
        for name, values in result["target_test"].items():
            print(f"{name:36} AUROC={values['auroc']:.4f} AUPRC={values['auprc']:.4f}", flush=True)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"settings":vars(args),"results":summary},
                                  indent=2), encoding="utf-8")
        print(f"Saved through {method}: {out}", flush=True)


if __name__ == "__main__":
    main()
