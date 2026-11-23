"""Report fixed-Xception detector performance on held-out test arrays.

This is detector REAL-vs-FAKE AUROC (FAKE is positive), NOT TRIAGE
error-detection AUROC (error is positive). Uses logits and labels only.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score


def compute(path):
    root = Path(path)
    y = np.asarray(np.load(root / "labels.npy", mmap_mode="r")).reshape(-1)
    z = np.asarray(np.load(root / "logits.npy", mmap_mode="r")).reshape(-1)
    if len(y) != len(z) or not len(y):
        raise ValueError(f"Bad array lengths: {root}")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(z).all():
        raise ValueError(f"Nonbinary labels or nonfinite logits: {root}")
    pred = (z >= 0).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    n = len(y)
    nreal, nfake = tn + fp, tp + fn
    return {
        "n": n, "real": nreal, "fake": nfake,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tn + tp) / n,
        "detector_auroc": float(roc_auc_score(y, z)) if nreal and nfake else None,
        "fpr": fp / nreal if nreal else None,
        "fnr": fn / nfake if nfake else None,
        "errors": fp + fn,
        "error_rate": (fp + fn) / n,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", action="append", nargs=2,
                   metavar=("NAME", "TEST_ARRAY_DIR"), required=True)
    p.add_argument("--output-json", default="results/detector_performance.json")
    p.add_argument("--output-csv", default="results/detector_performance.csv")
    args = p.parse_args()
    result = {name: compute(folder) for name, folder in args.dataset}
    for name, v in result.items():
        print(f"{name}: n={v['n']} (real={v['real']}, fake={v['fake']}), "
              f"TN={v['tn']}, FP={v['fp']}, FN={v['fn']}, TP={v['tp']}, "
              f"acc={v['accuracy']:.4f}, AUROC={v['detector_auroc']:.4f}, "
              f"FPR={v['fpr']:.4f}, FNR={v['fnr']:.4f}, "
              f"errors={v['errors']}, error_rate={v['error_rate']:.4f}")
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_out = Path(args.output_csv)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        fields = ["dataset"] + list(next(iter(result.values())))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, v in result.items():
            writer.writerow({"dataset": name, **v})
    print(f"Saved {output} and {csv_out}")


if __name__ == "__main__":
    main()
