"""Paired source-group bootstrap of error-ranking metric differences.

Both methods use identical resampled source groups in each bootstrap draw.
CSV must include group_id, error, baseline column, and candidate column.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def paired_differences(rows, baseline, candidate, repetitions=2000, seed=42):
    if repetitions < 1: raise ValueError("repetitions must be positive")
    groups=defaultdict(list)
    y=[]; a=[]; b=[]
    for i,row in enumerate(rows):
        if not row.get("group_id"): raise ValueError("Missing group_id")
        groups[row["group_id"]].append(i)
        y.append(int(row["error"]))
        a.append(float(row[baseline]))
        b.append(float(row[candidate]))
    y=np.asarray(y,dtype=int);a=np.asarray(a);b=np.asarray(b)
    if len(y)==0 or not np.isin(y,[0,1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Need nonempty binary error labels containing both classes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Scores must be finite")
    keys=list(groups)
    rng=np.random.default_rng(seed)
    metrics={"auroc":roc_auc_score,"auprc":average_precision_score}
    diffs={metric:[] for metric in metrics}
    for _ in range(repetitions):
        sampled=rng.choice(len(keys),size=len(keys),replace=True)
        idx=np.concatenate([groups[keys[j]] for j in sampled])
        if np.unique(y[idx]).size!=2: continue
        for name,func in metrics.items():
            diffs[name].append(float(func(y[idx],b[idx])-func(y[idx],a[idx])))
    if not diffs["auroc"]: raise ValueError("No usable bootstrap samples")
    result={"baseline":baseline,"candidate":candidate,"n_source_groups":len(keys),
            "n_rows":len(y),"valid_repetitions":len(diffs["auroc"]),"differences":{}}
    for name,fn in metrics.items():
        d=np.array(diffs[name])
        result["differences"][name]={
            "point":float(fn(y,b)-fn(y,a)),
            "ci95":np.quantile(d,[.025,.975]).tolist(),
            "fraction_bootstrap_positive":float(np.mean(d>0))}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",required=True)
    p.add_argument("--baseline",default="confidence_risk")
    p.add_argument("--candidate",default="Confidence_conditioned_TRIAGE")
    p.add_argument("--repetitions",type=int,default=2000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    with open(args.csv,newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    result=paired_differences(rows,args.baseline,args.candidate,args.repetitions,args.seed)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
