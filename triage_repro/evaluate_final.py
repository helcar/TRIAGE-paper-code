"""Final-paper TRIAGE misclassification ranking: six signals, VAL replay, TEST audit.

Single detector-agnostic evaluator for Xception, EfficientNet-B4 and ViT.
Uses the held-out TRIAGE split protocol FIT (prototypes/topics), CAL
(correct-prediction profiles), VAL (frozen config), TEST (report only).
Input arrays are extracted by modules in code/, with matching manifest rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from code.triage import Config, TRIAGE
from code.run_experiment import read, pred
from code.validate_triage import prototype_matrices
from code.evaluation import (
    FeatureBaselines, confidence_score, error_labels, evaluate,
    fit_prototype_profiles, prototype_attribution_divergence,
)


SIGNALS = ("Confidence", "Centroid", "Mahalanobis", "kNN", "PAD", "TRIAGE")
PARTITIONS = ("fit", "cal", "val", "test")


def source_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_split_manifest(manifest, require_official_ffpp=False):
    with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"path", "label", "split", "group_id", "video_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    by_split = {split: [] for split in PARTITIONS}
    groups = {}
    paths = set()
    for row in rows:
        split = row["split"]
        if split not in PARTITIONS or row["label"] not in {"0", "1"}:
            raise ValueError(f"Invalid split or label: {row}")
        gid = row["group_id"]
        if not gid:
            raise ValueError("Empty source-group ID")
        previous = groups.setdefault(gid, split)
        if previous != split:
            raise ValueError(f"Source-group leakage: {gid}")
        key = row["path"].replace("\\", "/").casefold()
        if key in paths:
            raise ValueError(f"Duplicate frame path: {key}")
        paths.add(key)
        if require_official_ffpp:
            expected = "train" if split in {"fit", "cal"} else split
            if row.get("official_split") != expected:
                raise ValueError(f"FF++ official split mismatch at {key}")
        by_split[split].append(row)
    if any(not by_split[k] for k in PARTITIONS):
        raise ValueError("Each FIT/CAL/VAL/TEST split must contain frames")
    return by_split


def load_arrays(root, split, rows, other_prov=None, manifest_sha=None):
    folder = Path(root) / split
    marker = folder / "_complete.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Incomplete extracted split: {folder}")
    meta = json.loads(marker.read_text(encoding="utf-8"))
    current = {key: meta.get(key) for key in
               ("checkpoint_sha256", "layer", "image_size", "normalization")}
    if current["checkpoint_sha256"] is None:
        raise ValueError(f"{folder}: missing detector checkpoint provenance")
    if other_prov is not None and current != other_prov:
        raise ValueError(f"{folder}: detector / layer / preprocessing provenance mismatch")
    if manifest_sha is not None and meta.get("manifest_sha256") not in (None, manifest_sha):
        raise ValueError(f"{folder}: wrong frame manifest hash")
    data = read(folder)
    expected_labels = np.asarray([int(r["label"]) for r in rows], dtype=int)
    if len(data["labels"]) != len(rows):
        raise ValueError(f"{split}: extracted images and manifest rows differ")
    if not np.array_equal(np.asarray(data["labels"]).reshape(-1), expected_labels):
        raise ValueError(f"{split}: extracted labels differ from ordered manifest rows")
    if not np.isfinite(data["logits"]).all():
        raise ValueError(f"{split}: nonfinite logits")
    return data, current


def config_from_selection(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    best = report["best"]["full_topic"]
    grid = report["search"]
    if best.get("kind") != "full_topic":
        raise ValueError("Selected candidate must be a complete TRIAGE topic model")
    cfg = Config(
        prototypes=int(best["K"]),
        topics_positive=int(best["M"]),
        topics_negative=int(best["M"]),
        references_real=int(best["J"]),
        references_fake=int(best["J"]),
        tau=float(best["tau"]),
        lambda_support=float(best["lambda_support"]),
        lambda_oppose=float(best["lambda_oppose"]),
        lambda_magnitude=float(best["lambda_magnitude"]),
        max_local_features=int(grid["max_local_features"]),
        seed=int(grid["seed"]),
    )
    return report, cfg


def pool(data, batch):
    return np.vstack([
        np.asarray(data["features"][i:i + batch]).mean(axis=1)
        for i in range(0, len(data["labels"]), batch)
    ])


def metric_subset(errors, confidence, signals):
    mask = {
        "overall": np.ones(len(errors), dtype=bool),
        "low_confidence_lt_0_70": confidence < 0.7,
        "high_confidence_ge_0_70": confidence >= 0.7,
    }
    return {
        name: {
            "frames": int(selection.sum()),
            "errors": int(errors[selection].sum()),
            "metrics": {key: evaluate(errors[selection], scores[selection])
                        for key, scores in signals.items()} if selection.any() else {},
        }
        for name, selection in mask.items()
    }


def review_queue(y, prediction, confidence, scores, budgets=(0.05, 0.1, 0.2)):
    errors = y != prediction
    fps = errors & (y == 0)
    fns = errors & (y == 1)
    output = {}
    for key, risk in scores.items():
        output[key] = {}
        for budget in budgets:
            n = max(1, int(np.ceil(budget * len(y))))
            # Stable tie-breaking by original manifest row order.
            chosen = np.argsort(risk, kind="stable")[-n:]
            output[key][str(budget)] = {
                "reviewed": n,
                "errors_found": int(errors[chosen].sum()),
                "false_positives_found": int(fps[chosen].sum()),
                "false_negatives_found": int(fns[chosen].sum()),
                "all_errors": int(errors.sum()),
                "all_false_positives": int(fps.sum()),
                "all_false_negatives": int(fns.sum()),
                "queue_precision": float(errors[chosen].mean()),
                "error_recall": float(errors[chosen].sum() / errors.sum()) if errors.any() else None,
            }
    return output


def group_bootstrap_auc_delta(errors, score_a, score_b, groups, draws=1000, seed=42):
    unique = np.asarray(sorted(set(groups)), dtype=object)
    lookup = {g: np.flatnonzero(np.asarray(groups) == g) for g in unique}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        group_sample = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([lookup[g] for g in group_sample])
        if len(set(errors[idx])) == 2:
            deltas.append(float(roc_auc_score(errors[idx], score_a[idx])
                                - roc_auc_score(errors[idx], score_b[idx])))
    return {
        "source_groups": len(unique),
        "replicates_requested": draws,
        "replicates_valid": len(deltas),
        "difference": float(roc_auc_score(errors, score_a)
                            - roc_auc_score(errors, score_b)) if len(set(errors)) == 2 else None,
        "ci95": np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else None,
    }


def run(args):
    by_split = read_split_manifest(args.manifest, args.ffpp_official)
    manifest_sha = source_sha(args.manifest)
    selection, cfg = config_from_selection(args.selection_json)
    provenance = None
    arrays = {}
    # TEST is intentionally NOT READ or fingerprinted prior to validation replay.
    for split in ("fit", "cal", "val"):
        data, provenance = load_arrays(
            args.arrays_root, split, by_split[split], provenance, manifest_sha)
        arrays[split] = data

    model = TRIAGE(cfg)
    batch = args.transform_batch
    fit, cal, val = (arrays[split] for split in ("fit", "cal", "val"))
    model.fit_prototypes(
        fit["features"][i:i + batch]
        for i in range(0, len(fit["labels"]), batch))
    f_positive, f_negative = prototype_matrices(model, fit, batch)
    c_positive, c_negative = prototype_matrices(model, cal, batch)
    model.fit_topics(f_positive, f_negative)
    model.fit_references(model.encode(c_positive, c_negative), cal["labels"], pred(cal))
    v_positive, v_negative = prototype_matrices(model, val, batch)
    val_errors = error_labels(val["labels"], pred(val))
    val_score = model.score(model.encode(v_positive, v_negative), pred(val))
    if len(np.unique(val_errors)) != 2:
        raise ValueError("Validation does not contain both correct and incorrect predictions")
    auc = float(roc_auc_score(val_errors, val_score))
    ap = float(average_precision_score(val_errors, val_score))
    best = selection["best"]["full_topic"]
    if abs(auc - best["auroc"]) > args.replay_tolerance or abs(ap - best["auprc"]) > args.replay_tolerance:
        raise RuntimeError(
            f"Validation replay mismatch; TEST untouched. "
            f"Expected {best['auroc']}/{best['auprc']}, got {auc}/{ap}")
    print(f"Validation replay VERIFIED: AUROC={auc:.6f}, AUPRC={ap:.6f}", flush=True)
    if not args.authorize_test:
        return {"validation_only": True, "auroc": auc, "auprc": ap}

    test, _ = load_arrays(args.arrays_root, "test", by_split["test"], provenance, manifest_sha)
    t_positive, t_negative = prototype_matrices(model, test, batch)
    cp = pred(test)
    errors = error_labels(test["labels"], cp)
    feature = FeatureBaselines().fit(pool(cal, batch), cal["labels"], pred(cal))
    features = feature.score(pool(test, batch), cp)
    pad_reference = fit_prototype_profiles(c_positive, c_negative, cal["labels"], pred(cal))
    signals = {
        "Confidence": confidence_score(test["logits"]),
        "Centroid": features["centroid"],
        "Mahalanobis": features["mahalanobis"],
        "kNN": features["knn"],
        "PAD": prototype_attribution_divergence(t_positive, t_negative, cp, pad_reference),
        "TRIAGE": model.score(model.encode(t_positive, t_negative), cp),
    }
    conf = 1.0 - signals["Confidence"]
    groups = [row["group_id"] for row in by_split["test"]]
    report = {
        "dataset": args.dataset, "model": args.model,
        "manifest_sha256": manifest_sha, "selection_sha256": source_sha(args.selection_json),
        "selection": best, "config": asdict(cfg), "provenance": provenance,
        "validation_replay": {"auroc": auc, "auprc": ap},
        "test_frames": len(errors), "test_errors": int(errors.sum()),
        "test_detector_accuracy": float(1.0 - errors.mean()),
        "results": metric_subset(errors, conf, signals),
        "review_queues": review_queue(np.asarray(test["labels"]), cp, conf, signals),
        "source_group_bootstrap_auc_triage_minus_confidence":
            group_bootstrap_auc_delta(errors, signals["TRIAGE"], signals["Confidence"],
                                      groups, args.bootstrap_repeats, cfg.seed),
        "note": "Frame-level outcomes. The test split was accessed only after VAL replay.",
    }
    output = Path(args.output_json)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite final report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.scores_csv:
        scores_path = Path(args.scores_csv)
        if scores_path.exists():
            raise FileExistsError(f"Refusing to overwrite final scores: {scores_path}")
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with scores_path.open("w", newline="", encoding="utf-8") as stream:
            columns = ("video_id", "group_id", "method", "label", "prediction",
                       "error", "confidence", *SIGNALS)
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for i, row in enumerate(by_split["test"]):
                writer.writerow({
                    "video_id": row["video_id"], "group_id": row["group_id"],
                    "method": row.get("method", ""), "label": int(test["labels"][i]),
                    "prediction": int(cp[i]), "error": int(errors[i]),
                    "confidence": float(conf[i]),
                    **{name: float(values[i]) for name, values in signals.items()},
                })
    print(json.dumps({"model": args.model, "dataset": args.dataset,
                      "frames": len(errors), "errors": int(errors.sum()),
                      "AUROC": {name: value["auroc"] for name, value
                                in report["results"]["overall"]["metrics"].items()}}, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--arrays-root", required=True)
    p.add_argument("--selection-json", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, choices=("xception", "efficientnetb4", "vit"))
    p.add_argument("--output-json", required=True)
    p.add_argument("--scores-csv")
    p.add_argument("--transform-batch", type=int, default=64)
    p.add_argument("--bootstrap-repeats", type=int, default=1000)
    p.add_argument("--replay-tolerance", type=float, default=1e-6)
    p.add_argument("--ffpp-official", action="store_true")
    p.add_argument("--authorize-test", action="store_true")
    args = p.parse_args()
    if args.transform_batch < 1 or args.bootstrap_repeats < 1:
        p.error("Batch and bootstrap repeats must be positive")
    run(args)


if __name__ == "__main__":
    main()
