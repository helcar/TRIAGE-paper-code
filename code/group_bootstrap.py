"""Paired group bootstrap on SAVED errors and model risk scores.

CSV columns: group_id,error,TRIAGE,Confidence[,other scores].
Multiple rows from one linked source group are always resampled together.
No video decoding, detector inference, or TRIAGE fitting is performed.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def bootstrap(rows, score_columns, repetitions=1000, seed=42):
    if not rows: raise ValueError("Empty CSV")
    groups={}
    for i,row in enumerate(rows):
        if not row.get("group_id"): raise ValueError("group_id required")
        groups.setdefault(row["group_id"],[]).append(i)
    y=np.array([int(r["error"]) for r in rows])
    if not set(np.unique(y)).issubset({0,1}):
        raise ValueError("error must be binary")
    scores={name:np.array([float(r[name]) for r in rows]) for name in score_columns}
    if any(not np.isfinite(s).all() for s in scores.values()):
        raise ValueError("Nonfinite score")
    rng=np.random.default_rng(seed)
    keys=list(groups)
    summary={name:{"auroc":[],"auprc":[]} for name in score_columns}
    for _ in range(repetitions):
        sample=rng.choice(keys,size=len(keys),replace=True)
        idx=np.concatenate([groups[key] for key in sample])
        if len(np.unique(y[idx]))<2: continue
        for name,s in scores.items():
            summary[name]["auroc"].append(roc_auc_score(y[idx],s[idx]))
            summary[name]["auprc"].append(average_precision_score(y[idx],s[idx]))
    result={"n_source_groups":len(keys),"n_rows":len(y),
            "valid_repetitions":len(next(iter(summary.values()))["auroc"]),
            "metrics":{}}
    if not result["valid_repetitions"]:
        raise ValueError("No bootstrap replicates contained both outcome classes")
    for name, metrics in summary.items():
        result["metrics"][name]={
            key:{"point":float(func(y,scores[name])),
                 "ci95":np.quantile(values,[.025,.975]).tolist()}
            for key, values,func in (("auroc",metrics["auroc"],roc_auc_score),
                                    ("auprc",metrics["auprc"],average_precision_score))}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",required=True)
    p.add_argument("--scores",nargs="+",default=["TRIAGE","Confidence"])
    p.add_argument("--repetitions",type=int,default=1000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    with open(args.csv,newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    report=bootstrap(rows,args.scores,args.repetitions,args.seed)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
