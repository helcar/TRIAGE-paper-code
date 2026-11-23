"""Validation-only layer ablation over existing per-layer extracted arrays.

One fixed TRIAGE configuration, independent per-layer fit/cal, same val rows.
No test-set access. Uses memmap-backed arrays and bounded prototype reservoir.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .triage import Config
from .compare_layers import fit_layer
from .evaluation import confidence_score, error_labels
from .run_experiment import pred


def review_recall(errors,scores,percent):
    k=max(1,int(np.ceil(len(errors)*percent/100)))
    idx=np.argsort(-scores,kind="stable")[:k]
    return {"reviewed":k,"errors_found":int(errors[idx].sum()),
            "error_recall":float(errors[idx].sum()/errors.sum())}


def metrics(y,s):
    out={"AUROC":float(roc_auc_score(y,s)),
         "AUPRC":float(average_precision_score(y,s))}
    out["review"]={str(p)+"pct":review_recall(y,s,p) for p in (5,10,20)}
    return out


def check_alignment(reference, other, layer, split, atol=0.005, rtol=1e-5):
    """Compare stable labels, decisions, and approximately equal detector logits."""
    a, b = reference[split], other[split]
    if not np.array_equal(a["labels"], b["labels"]):
        raise ValueError(f"Layer {layer} has different labels/order in {split}")
    x = np.asarray(a["logits"], dtype=np.float64)
    y = np.asarray(b["logits"], dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Layer {layer} logits shape mismatch in {split}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"Layer {layer} has nonfinite logits in {split}")
    mismatch = int(np.count_nonzero((x >= 0) != (y >= 0)))
    max_diff = float(np.max(np.abs(x-y))) if len(x) else 0.
    if mismatch or not np.allclose(x, y, rtol=rtol, atol=atol):
        raise ValueError(
            f"Layer {layer} does not align on {split}/logits: "
            f"prediction_mismatches={mismatch}, max_abs_diff={max_diff:.6g}, "
            f"atol={atol}, rtol={rtol}. Check sample order, checkpoint, "
            "preprocessing, and detector extraction settings.")
    return {"max_abs_logit_difference":max_diff, "prediction_mismatches":mismatch}


def run_layers(roots,cfg,batch,logit_atol=0.005):
    all_data={};results={};scores={}
    for name,root in roots.items():
        data,cal_score,val_score=fit_layer(root,cfg,batch)
        all_data[name]=data
        scores[name]=val_score
        results[name]={"feature_shape":list(data["val"]["features"].shape),
                       "metrics":None}
    names=list(roots)
    reference=all_data[names[0]]
    alignment={}
    for name in names[1:]:
        alignment[name]={}
        for split in ("fit","cal","val"):
            alignment[name][split]=check_alignment(reference,all_data[name],name,split,atol=logit_atol)
    val=reference["val"]
    y=error_labels(val["labels"],pred(val))
    if np.unique(y).size!=2:raise ValueError("Validation requires both outcomes")
    for name in names: results[name]["metrics"]=metrics(y,scores[name])
    confidence=confidence_score(val["logits"])
    return {"validation_images":int(len(y)),"validation_errors":int(y.sum()),
            "config":{"K":cfg.prototypes,"M_positive":cfg.topics_positive,
                      "M_negative":cfg.topics_negative,
                      "J_real":cfg.references_real,"J_fake":cfg.references_fake},
            "layers":results,"alignment":alignment,"confidence":metrics(y,confidence)},{
                **scores,"confidence":confidence,"errors":y}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer",action="append",required=True,metavar="NAME=DIR",
                   help="Repeat for each extracted feature layer")
    p.add_argument("--output",required=True)
    p.add_argument("--prototypes",type=int,default=128)
    p.add_argument("--topics",type=int,default=16)
    p.add_argument("--references",type=int,default=2)
    p.add_argument("--max-local-features",type=int,default=50000)
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--logit-atol",type=float,default=0.005,
                   help="Maximum absolute tolerance for cross-extraction logit differences; all predictions must match")
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    roots={}
    for spec in args.layer:
        if "=" not in spec: p.error("--layer must be NAME=DIR")
        name,path=spec.split("=",1)
        if not name or not path or name in roots:p.error("Missing/duplicate layer name")
        roots[name]=path
    cfg=Config(prototypes=args.prototypes,topics_positive=args.topics,
               topics_negative=args.topics,references_real=args.references,
               references_fake=args.references,max_local_features=args.max_local_features,
               seed=args.seed)
    if args.logit_atol < 0: p.error("--logit-atol must be nonnegative")
    report,scores=run_layers(roots,cfg,args.batch,args.logit_atol)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    np.savez_compressed(out.with_name(out.stem+"_scores.npz"),**scores)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
