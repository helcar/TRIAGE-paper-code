"""Compare two TRIAGE layers and a calibration-normalized late fusion on validation.

Uses cached feature arrays; no test data or learned fusion weights.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .triage import TRIAGE, Config
from .run_experiment import read, pred
from .validate_triage import prototype_matrices
from .evaluation import confidence_score, error_labels


def empirical_percentile(reference, samples):
    """Map scores to calibration empirical CDF, robust to unequal layer scales."""
    ref=np.sort(np.asarray(reference,dtype=float))
    if not len(ref) or not np.isfinite(ref).all() or not np.isfinite(samples).all():
        raise ValueError("Scores must be nonempty and finite")
    return np.searchsorted(ref,samples,side="right")/len(ref)


def fit_layer(splits,cfg,batch):
    d={key:read(Path(splits)/key) for key in ("fit","cal","val")}
    model=TRIAGE(cfg)
    fit=d["fit"]
    model.fit_prototypes(fit["features"][i:i+batch]
                         for i in range(0,len(fit["labels"]),batch))
    fp=prototype_matrices(model,fit,batch)
    cp=prototype_matrices(model,d["cal"],batch)
    vp=prototype_matrices(model,d["val"],batch)
    model.fit_topics(*fp)
    cal=d["cal"]
    ce=model.encode(*cp)
    model.fit_references(ce,cal["labels"],pred(cal))
    return d,model.score(ce,pred(cal)),model.score(model.encode(*vp),pred(d["val"]))


def evaluate(errors,s):
    return {"AUROC":float(roc_auc_score(errors,s)),
            "AUPRC":float(average_precision_score(errors,s))}


def compare(layer_a,layer_b,cfg,batch=64):
    a,cal_a,val_a=fit_layer(layer_a,cfg,batch)
    b,cal_b,val_b=fit_layer(layer_b,cfg,batch)
    for split in ("fit","cal","val"):
        for field in ("logits","labels"):
            if not np.array_equal(a[split][field],b[split][field]):
                raise ValueError(f"Different manifest order or detector outputs: {split}/{field}")
    errors=error_labels(a["val"]["labels"],pred(a["val"]))
    if np.unique(errors).size!=2:
        raise ValueError("Validation must contain correct and incorrect predictions")
    normalized_a=empirical_percentile(cal_a,val_a)
    normalized_b=empirical_percentile(cal_b,val_b)
    fusion=(normalized_a+normalized_b)/2
    confidence=confidence_score(a["val"]["logits"])
    return {"validation_images":int(len(errors)),"validation_errors":int(errors.sum()),
            "fusion":"Equal-weight mean of empirical calibration-CDF scores, no weights fitted on validation",
            "metrics":{"Layer_A":evaluate(errors,val_a),
                       "Layer_B":evaluate(errors,val_b),
                       "Equal_weight_two_layer_fusion":evaluate(errors,fusion),
                       "Confidence":evaluate(errors,confidence)},
            "scores":{"a":val_a,"b":val_b,"fusion":fusion,"confidence":confidence,
                      "errors":errors}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("layer-a","layer-b","output"):p.add_argument("--"+name,required=True)
    p.add_argument("--prototypes",type=int,default=128)
    p.add_argument("--topics",type=int,default=16)
    p.add_argument("--references",type=int,default=2)
    p.add_argument("--max-local-features",type=int,default=50000)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    cfg=Config(prototypes=args.prototypes,topics_positive=args.topics,
               topics_negative=args.topics,references_real=args.references,
               references_fake=args.references,max_local_features=args.max_local_features,
               seed=args.seed)
    result=compare(args.layer_a,args.layer_b,cfg,args.batch)
    scores=result.pop("scores")
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez_compressed(out.with_name(out.stem+"_scores.npz"),**scores)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
