"""Validation-only TRIAGE pilot using cached Xception outputs.

No test split is opened. Select hyperparameters using validation, then freeze
configurations before evaluating official FF++ test data.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .triage import TRIAGE, Config
from .run_experiment import read, pred
from .evaluation import error_labels, confidence_score


def prototype_matrices(model, data, batch=64):
    positives, negatives = [], []
    for i in range(0, len(data["labels"]), batch):
        pos, neg = model.transform_prototypes(
            data["features"][i:i+batch], data["attributions"][i:i+batch])
        positives.append(pos)
        negatives.append(neg)
    return np.vstack(positives), np.vstack(negatives)


def run(fit_dir, cal_dir, val_dir, cfg, transform_batch=64):
    fit, cal, val = (read(path) for path in (fit_dir,cal_dir,val_dir))
    model=TRIAGE(cfg)
    # fit_prototypes internally uses a bounded reservoir; read memmap in blocks
    model.fit_prototypes(
        fit["features"][i:i+transform_batch]
        for i in range(0,len(fit["labels"]),transform_batch)
    )
    fp=prototype_matrices(model,fit,transform_batch)
    cp=prototype_matrices(model,cal,transform_batch)
    vp=prototype_matrices(model,val,transform_batch)
    model.fit_topics(*fp)
    model.fit_references(model.encode(*cp),cal["labels"],pred(cal))
    y=error_labels(val["labels"],pred(val))
    if len(np.unique(y))!=2:
        raise ValueError("Validation has no correct/error mixture")
    score=model.score(model.encode(*vp),pred(val))
    conf=confidence_score(val["logits"])
    return {"config":asdict(cfg),
            "validation_images":len(y),
            "validation_errors":int(y.sum()),
            "TRIAGE":{"AUROC":float(roc_auc_score(y,score)),
                      "AUPRC":float(average_precision_score(y,score))},
            "Confidence":{"AUROC":float(roc_auc_score(y,conf)),
                          "AUPRC":float(average_precision_score(y,conf))}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("fit","cal","val","output"):p.add_argument("--"+key,required=True)
    p.add_argument("--prototypes",type=int,default=128)
    p.add_argument("--topics",type=int,default=16)
    p.add_argument("--references",type=int,default=2)
    p.add_argument("--max-local-features",type=int,default=50000)
    p.add_argument("--transform-batch",type=int,default=64)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    cfg=Config(prototypes=args.prototypes,topics_positive=args.topics,
               topics_negative=args.topics,references_real=args.references,
               references_fake=args.references,max_local_features=args.max_local_features,
               seed=args.seed)
    result=run(args.fit,args.cal,args.val,cfg,args.transform_batch)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
