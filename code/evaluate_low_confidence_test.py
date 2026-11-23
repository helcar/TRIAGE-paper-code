"""Frozen test subgroup analysis: confidence < 70%, no tuning or refitting.

Consumes the per-frame CSV produced by evaluate_frozen_triage.py.
Resamples connected source-video groups for paired confidence intervals.
This script must not be used to choose a new threshold after observing test.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .paired_group_bootstrap import paired_differences


def result(rows, baseline="Confidence",candidate="TRIAGE",repetitions=2000,seed=42):
    if not rows: raise ValueError("Empty test CSV")
    # Confidence is stored as 1 - max class probability, hence <70% confidence
    # corresponds to uncertainty > 0.30 (strict inequality).
    selected=[r for r in rows if float(r[baseline]) > .30]
    if not selected: raise ValueError("No confidence < 70% observations")
    y=np.array([int(r["error"]) for r in selected])
    if np.unique(y).size!=2:
        raise ValueError("Low-confidence subset must have correct and incorrect predictions")
    def metric(name):
        s=np.array([float(r[name]) for r in selected])
        return {"auroc":float(roc_auc_score(y,s)),
                "auprc":float(average_precision_score(y,s))}
    paired=paired_differences(selected,baseline,candidate,repetitions,seed)
    return {"threshold":"maximum class probability < 0.70 (fixed from development)",
            "test_frames":len(rows),"low_confidence_frames":len(selected),
            "low_confidence_source_groups":len({r["group_id"] for r in selected}),
            "low_confidence_errors":int(y.sum()),
            "low_confidence_error_prevalence":float(y.mean()),
            baseline:metric(baseline),candidate:metric(candidate),
            "paired_candidate_minus_baseline":paired["differences"],
            "bootstrap_valid_repetitions":paired["valid_repetitions"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--repetitions",type=int,default=2000)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    with open(a.csv,newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    report=result(rows,repetitions=a.repetitions,seed=a.seed)
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
