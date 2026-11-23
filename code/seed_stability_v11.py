"""B5: frozen-config TRIAGE stability across seeds on validation and held-out test.

The same selected hyperparameters are used at every seed. This is a
sensitivity experiment, not seed selection; do not choose a new test seed.
Read-only FIT/CAL/VAL/TEST arrays; no detector inference or optimization.
"""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from .triage import TRIAGE, Config
from .run_experiment import read, pred
from .validate_triage import prototype_matrices
from .evaluation import confidence_score, error_labels


def met(y, score):
    if len(np.unique(y)) != 2:
        raise ValueError("Both errors and correct predictions are required")
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score))}


def fit_score(cfg, fit, cal, val, test, batch):
    model = TRIAGE(cfg)
    model.fit_prototypes(fit["features"][i:i + batch]
                         for i in range(0, len(fit["labels"]), batch))
    fp = prototype_matrices(model, fit, batch)
    cp = prototype_matrices(model, cal, batch)
    model.fit_topics(*fp)
    model.fit_references(model.encode(*cp), cal["labels"], pred(cal))
    vp = prototype_matrices(model, val, batch)
    tp = prototype_matrices(model, test, batch)
    return (model.score(model.encode(*vp), pred(val)),
            model.score(model.encode(*tp), pred(test)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("fit", "cal", "val", "test", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 42])
    p.add_argument("--prototypes", type=int, default=512)
    p.add_argument("--topics", type=int, default=4)
    p.add_argument("--references", type=int, default=2)
    p.add_argument("--tau", type=float, default=.01)
    p.add_argument("--lambda-support", type=float, default=1.)
    p.add_argument("--lambda-oppose", type=float, default=0.)
    p.add_argument("--lambda-magnitude", type=float, default=.1)
    p.add_argument("--max-local-features", type=int, default=50000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--frozen-scores-csv",
                   help="Optional original seed-42 test CSV; verifies unchanged test scoring")
    p.add_argument("--tolerance", type=float, default=1e-5)
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds) or len(a.seeds) < 2:
        p.error("Provide at least two distinct seeds")
    if a.batch < 1:
        p.error("--batch must be positive")
    fit, cal, val, test = [read(getattr(a, s)) for s in ("fit", "cal", "val", "test")]
    vy = error_labels(val["labels"], pred(val))
    ty = error_labels(test["labels"], pred(test))
    report = {
        "protocol": "fixed hyperparameters across seeds; all seeds evaluated, no test selection",
        "configuration": {"K": a.prototypes, "M": a.topics, "J": a.references,
                          "tau": a.tau, "lambda_support": a.lambda_support,
                          "lambda_oppose": a.lambda_oppose,
                          "lambda_magnitude": a.lambda_magnitude},
        "validation_count": int(len(vy)), "validation_errors": int(vy.sum()),
        "test_count": int(len(ty)), "test_errors": int(ty.sum()),
        "confidence": {"validation": met(vy, confidence_score(val["logits"])),
                       "test": met(ty, confidence_score(test["logits"]))},
        "seeds": {}, "summary": {},
    }
    frozen_rows = None
    if a.frozen_scores_csv:
        with open(a.frozen_scores_csv, newline="", encoding="utf-8-sig") as f:
            frozen_rows = list(csv.DictReader(f))
        if len(frozen_rows) != len(ty) or "TRIAGE" not in frozen_rows[0]:
            raise ValueError("Frozen CSV is missing aligned TRIAGE test scores")
        if not np.array_equal(ty, [int(x["error"]) for x in frozen_rows]):
            raise ValueError("Frozen CSV test-error alignment fails")
    for seed in a.seeds:
        cfg = Config(prototypes=a.prototypes, topics_positive=a.topics,
                     topics_negative=a.topics, references_real=a.references,
                     references_fake=a.references, tau=a.tau,
                     lambda_support=a.lambda_support,
                     lambda_oppose=a.lambda_oppose,
                     lambda_magnitude=a.lambda_magnitude,
                     max_local_features=a.max_local_features, seed=seed)
        vs, ts = fit_score(cfg, fit, cal, val, test, a.batch)
        if frozen_rows is not None and seed == 42:
            previous = np.array([float(x["TRIAGE"]) for x in frozen_rows])
            if not np.allclose(previous, ts, atol=a.tolerance, rtol=0):
                raise RuntimeError("Seed-42 TEST differs from frozen scores. Verify configuration.")
        report["seeds"][str(seed)] = {"validation": met(vy, vs),
                                     "test": met(ty, ts)}
        print(f"seed={seed}: VAL AUROC={report['seeds'][str(seed)]['validation']['auroc']:.4f} "
              f"AUPRC={report['seeds'][str(seed)]['validation']['auprc']:.4f} | "
              f"TEST AUROC={report['seeds'][str(seed)]['test']['auroc']:.4f} "
              f"AUPRC={report['seeds'][str(seed)]['test']['auprc']:.4f}", flush=True)
    for split in ("validation", "test"):
        report["summary"][split] = {}
        for metric in ("auroc", "auprc"):
            values = np.array([report["seeds"][str(seed)][split][metric]
                               for seed in a.seeds])
            report["summary"][split][metric] = {
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        print(f"{split.upper()} mean±SD: AUROC "
              f"{report['summary'][split]['auroc']['mean']:.4f}±"
              f"{report['summary'][split]['auroc']['sample_sd']:.4f}, AUPRC "
              f"{report['summary'][split]['auprc']['mean']:.4f}±"
              f"{report['summary'][split]['auprc']['sample_sd']:.4f}")
    clean_output = a.output.strip()
    if not clean_output:
        raise ValueError("Output filename is empty after trimming whitespace")
    if clean_output != a.output:
        print(f"Note: removed leading/trailing whitespace from output path: {a.output!r}")
    path = Path(clean_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
