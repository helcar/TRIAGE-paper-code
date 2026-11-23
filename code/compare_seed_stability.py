"""Compare two frozen TRIAGE configurations under identical validation seeds.

Inputs are results from code.evaluate_seed_stability on the same dataset.
No test data is read; no configuration is fitted or retrained.
This is a descriptive 4-or-more-seed comparison, NOT a significance test.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def load(path):
    with open(path,encoding="utf-8") as f:
        data=json.load(f)
    runs=data.get("seeds", [])
    if not runs:
        raise ValueError(f"{path}: no seed results")
    by_seed={int(r["seed"]):r for r in runs}
    if len(by_seed)!=len(runs):
        raise ValueError(f"{path}: duplicate seed results")
    if any("validation_auroc" not in r or "validation_auprc" not in r
           for r in runs):
        raise ValueError(f"{path}: missing validation metrics")
    return data,by_seed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous",required=True,
                   help="Seed-stability JSON using the PREVIOUS smaller-grid configuration")
    p.add_argument("--expanded",required=True,
                   help="Seed-stability JSON using EXPANDED K/M configuration")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    prev, a=load(args.previous)
    expanded, b=load(args.expanded)
    if set(a)!=set(b):
        raise ValueError(f"Seed sets mismatch: previous={sorted(a)}, expanded={sorted(b)}")
    if prev.get("validation_frames")!=expanded.get("validation_frames") or (
       prev.get("validation_errors")!=expanded.get("validation_errors")):
        raise ValueError("Validation partitions differ")
    n=len(a)
    if n<3:
        raise ValueError("At least three matched seeds needed")
    report={
        "protocol":"paired initialization seed diagnostics on VAL, NO test reads",
        "previous_config":prev.get("config",{}),
        "expanded_config":expanded.get("config",{}),
        "validation_frames":prev.get("validation_frames"),
        "validation_errors":prev.get("validation_errors"),
        "seed_count":n,
        "per_seed":[],
        "summary":{}
    }
    print("Previous vs expanded configuration, same validation seeds")
    for seed in sorted(a):
        left,right=a[seed],b[seed]
        row={"seed":seed}
        for metric in ("auroc","auprc"):
            key="validation_"+metric
            row["previous_"+metric]=float(left[key])
            row["expanded_"+metric]=float(right[key])
            row["expanded_minus_previous_"+metric]=float(right[key]-left[key])
        report["per_seed"].append(row)
        print(f"  Seed {seed}: AUROC previous={row['previous_auroc']:.4f}, "
              f"expanded={row['expanded_auroc']:.4f}, "
              f"delta={row['expanded_minus_previous_auroc']:+.4f}; "
              f"AUPRC delta={row['expanded_minus_previous_auprc']:+.4f}")
    for metric in ("auroc","auprc"):
        old=np.asarray([r["previous_"+metric] for r in report["per_seed"]])
        new=np.asarray([r["expanded_"+metric] for r in report["per_seed"]])
        diff=new-old
        report["summary"][metric]={
            "previous_mean":float(old.mean()),
            "expanded_mean":float(new.mean()),
            "mean_paired_delta":float(diff.mean()),
            "sample_std_of_paired_delta":float(diff.std(ddof=1)),
            "expanded_better_seed_count":int((diff>0).sum()),
            "equal_seed_count":int((diff==0).sum()),
            "matched_seed_count":n,
            "delta_min":float(diff.min()),
            "delta_max":float(diff.max()),
        }
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\nDescriptive comparison (no significance test):")
    print(json.dumps(report["summary"],indent=2))
    print(f"Saved {target}")


if __name__=="__main__":
    main()
