"""Freeze detector-training/holdout groups within TRIAGE FIT, like the ViT script."""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from .manifest import audit, load_manifest, sha256, write_rows


def partition(rows, seed=42, holdout_fraction=0.15):
    if not 0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be in (0, 0.5)")
    fit = [r for r in rows if r["split"] == "fit"]
    grouped = defaultdict(list)
    for row in fit:
        grouped[row["group_id"]].append(row)
    groups = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = max(1, round(len(groups) * holdout_fraction))
    if n >= len(groups):
        raise ValueError("Not enough groups for detector training and holdout")
    held = set(groups[:n])
    train = [r for r in fit if r["group_id"] not in held]
    holdout = [r for r in fit if r["group_id"] in held]
    if {r["group_id"] for r in train} & {r["group_id"] for r in holdout}:
        raise RuntimeError("Detector holdout group leakage")
    for name, subset in (("train", train), ("holdout", holdout)):
        if {r["label"] for r in subset} != {"0", "1"}:
            raise ValueError(f"{name} missing REAL or FAKE class")
    return train, holdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-fraction", type=float, default=0.15)
    a = p.parse_args()
    path = Path(a.manifest)
    rows = load_manifest(path)
    full_audit = audit(rows)
    train, holdout = partition(rows, a.seed, a.holdout_fraction)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, subset in (("detector_train.csv", train), ("detector_holdout.csv", holdout)):
        target = out / filename
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite frozen split: {target}")
        write_rows(target, subset)
    report = {
        "manifest_sha256": sha256(path),
        "seed": a.seed,
        "holdout_fraction": a.holdout_fraction,
        "detector_train_frames": len(train),
        "detector_holdout_frames": len(holdout),
        "detector_train_groups": len({r["group_id"] for r in train}),
        "detector_holdout_groups": len({r["group_id"] for r in holdout}),
        "train_labels": dict(Counter(r["label"] for r in train)),
        "holdout_labels": dict(Counter(r["label"] for r in holdout)),
        "manifest_audit": full_audit,
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
