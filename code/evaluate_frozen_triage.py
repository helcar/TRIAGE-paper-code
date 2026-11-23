"""Reproduce frozen selected TRIAGE on validation, then evaluate held-out test.

No tuning. The test directory is not read until the validation reproduction
matches --expected-val-auroc and --expected-val-auprc (if supplied).
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .triage import TRIAGE, Config
from .run_experiment import read, pred
from .evaluation import evaluate, error_labels, confidence_score
from .validate_triage import prototype_matrices
from .validate_confidence_conditioning import load_validation_groups


def run(a):
    fit,cal,val=(read(p) for p in (a.fit,a.cal,a.val))
    cfg=Config(prototypes=a.prototypes,topics_positive=a.topics,
        topics_negative=a.topics,references_real=a.references,
        references_fake=a.references,tau=a.tau,
        lambda_support=a.lambda_support,lambda_oppose=a.lambda_oppose,
        lambda_magnitude=a.lambda_magnitude,
        max_local_features=a.max_local_features,seed=a.seed)
    model=TRIAGE(cfg);batch=a.transform_batch
    model.fit_prototypes(fit["features"][i:i+batch]
                         for i in range(0,len(fit["labels"]),batch))
    fp=prototype_matrices(model,fit,batch)
    cp=prototype_matrices(model,cal,batch)
    model.fit_topics(*fp)
    model.fit_references(model.encode(*cp),cal["labels"],pred(cal))
    vp=prototype_matrices(model,val,batch)
    val_score=model.score(model.encode(*vp),pred(val))
    val_metrics=evaluate(error_labels(val["labels"],pred(val)),val_score)
    validation={"auroc":val_metrics["auroc"],"auprc":val_metrics["auprc"]}
    print("Validation reproduction:",json.dumps(validation),flush=True)
    if a.expected_val_auroc is not None and abs(validation["auroc"]-a.expected_val_auroc)>a.tolerance:
        raise RuntimeError("Validation AUROC mismatch; test has NOT been read")
    if a.expected_val_auprc is not None and abs(validation["auprc"]-a.expected_val_auprc)>a.tolerance:
        raise RuntimeError("Validation AUPRC mismatch; test has NOT been read")
    if not a.evaluate_test:
        return {"config":vars(a),"validation_reproduction":validation,
                "test_evaluated":False}
    test=read(a.test)
    with open(a.manifest,newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    original_val=[r for r in rows if r["split"]=="val"]
    original_test=[r for r in rows if r["split"]=="test"]
    if len(original_val)!=len(val["labels"]) or len(original_test)!=len(test["labels"]):
        raise ValueError("Manifest/arrays length mismatch")
    if any(int(r["label"])!=int(y) for r,y in zip(original_test,test["labels"])):
        raise ValueError("Manifest/test label order mismatch")
    if any(int(r["label"])!=int(y) for r,y in zip(original_val,val["labels"])):
        raise ValueError("Manifest/validation label order mismatch")
    if ({r["group_id"] for r in original_val} &
        {r["group_id"] for r in original_test}):
        raise ValueError("Validation and test source-group overlap")
    tp=prototype_matrices(model,test,batch)
    score=model.score(model.encode(*tp),pred(test))
    conf=confidence_score(test["logits"])
    err=error_labels(test["labels"],pred(test))
    results={"config":{"prototypes":cfg.prototypes,"topics":cfg.topics_positive,
        "references":cfg.references_real,"tau":cfg.tau,
        "lambda_support":cfg.lambda_support,"lambda_oppose":cfg.lambda_oppose,
        "lambda_magnitude":cfg.lambda_magnitude,"seed":cfg.seed,
        "max_local_features":cfg.max_local_features},
        "validation_reproduction":validation,
        "test_evaluated":True,
        "test":{"n":len(err),"errors":int(err.sum()),
                "TRIAGE":evaluate(err,score),
                "Confidence":evaluate(err,conf)}}
    if a.scores_csv:
        dest=Path(a.scores_csv);dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f)
            writer.writerow(["group_id","video_id","method","error","TRIAGE","Confidence"])
            for i,r in enumerate(original_test):
                writer.writerow([r["group_id"],r["video_id"],r["method"],int(err[i]),
                                 float(score[i]),float(conf[i])])
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for s in ("fit","cal","val","manifest","output"):
        p.add_argument("--"+s,required=True)
    p.add_argument("--test",help="Only used together with --evaluate-test")
    p.add_argument("--evaluate-test",action="store_true")
    p.add_argument("--scores-csv")
    p.add_argument("--expected-val-auroc",type=float)
    p.add_argument("--expected-val-auprc",type=float)
    p.add_argument("--tolerance",type=float,default=1e-5)
    p.add_argument("--prototypes",type=int,default=128)
    p.add_argument("--topics",type=int,default=8)
    p.add_argument("--references",type=int,default=1)
    p.add_argument("--tau",type=float,default=100)
    p.add_argument("--lambda-support",type=float,default=.01)
    p.add_argument("--lambda-oppose",type=float,default=.01)
    p.add_argument("--lambda-magnitude",type=float,default=1)
    p.add_argument("--max-local-features",type=int,default=50000)
    p.add_argument("--transform-batch",type=int,default=64)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.evaluate_test and not a.test:
        p.error("--test required with --evaluate-test")
    result=run(a)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
