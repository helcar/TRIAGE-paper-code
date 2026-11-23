"""B7 frozen ViT TEST evaluation; replay validation selection before opening TEST.

Exactly reproduce the bounded prototype sample and transform batch of the
validation-only search; forbid different configs or automatic reselection.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .triage import TRIAGE, Config
from .run_experiment import read, pred
from .validate_triage import prototype_matrices
from .evaluation import error_labels, confidence_score, evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("fit","cal","val","test","selection-json","output"):
        p.add_argument("--"+flag,required=True)
    p.add_argument("--authorize-test",action="store_true")
    p.add_argument("--transform-batch",type=int,default=32)
    p.add_argument("--max-local-features",type=int,default=50000)
    a=p.parse_args()
    if not a.authorize_test:
        p.error("Explicit --authorize-test required")
    report=json.loads(Path(a.selection_json).read_text(encoding="utf-8"))
    best=report["best"]["full_topic"]
    search=report["search"]
    if a.max_local_features != search["max_local_features"] or a.transform_batch != 32:
        p.error("Must replay validation with 50,000 features and batch 32")
    if best["kind"]!="full_topic" or search["seed"]!=42:
        p.error("Unexpected selection format or seed")
    cfg=Config(prototypes=best["K"],topics_positive=best["M"],
               topics_negative=best["M"],references_real=best["J"],
               references_fake=best["J"],tau=best["tau"],
               lambda_support=best["lambda_support"],
               lambda_oppose=best["lambda_oppose"],
               lambda_magnitude=best["lambda_magnitude"],
               max_local_features=a.max_local_features,seed=42)
    fit,cal,val=(read(q) for q in (a.fit,a.cal,a.val))
    model=TRIAGE(cfg)
    model.fit_prototypes(fit["features"][i:i+a.transform_batch]
                         for i in range(0,len(fit["labels"]),a.transform_batch))
    fp=prototype_matrices(model,fit,a.transform_batch)
    cp=prototype_matrices(model,cal,a.transform_batch)
    vp=prototype_matrices(model,val,a.transform_batch)
    model.fit_topics(*fp)
    model.fit_references(model.encode(*cp),cal["labels"],pred(cal))
    yval=error_labels(val["labels"],pred(val))
    vs=model.score(model.encode(*vp),pred(val))
    vu=float(roc_auc_score(yval,vs))
    va=float(average_precision_score(yval,vs))
    if abs(vu-best["auroc"])>1e-7 or abs(va-best["auprc"])>1e-7:
        raise RuntimeError(f"Validation replay mismatch: AUROC {vu} vs "
                           f"{best['auroc']}, AUPRC {va} vs {best['auprc']}; "
                           "TEST not opened.")
    print(f"Validation replay VERIFIED: AUROC={vu:.8f}, AUPRC={va:.8f}",flush=True)
    # Read TEST only after reproducing selection exactly.
    test=read(a.test)
    tp=prototype_matrices(model,test,a.transform_batch)
    ty=error_labels(test["labels"],pred(test))
    triage=model.score(model.encode(*tp),pred(test))
    confidence=confidence_score(test["logits"])
    result={"selection":best,"config":asdict(cfg),
            "validation_replay":{"n":len(yval),"errors":int(yval.sum()),
                                 "auroc":vu,"auprc":va},
            "test_detector":{"images":len(ty),"errors":int(ty.sum()),
                             "accuracy":float(np.mean(pred(test)==test["labels"])),
                             "auroc":float(roc_auc_score(test["labels"],test["logits"]))},
            "test_error_ranking":{"TRIAGE":evaluate(ty,triage),
                                  "Confidence":evaluate(ty,confidence)}}
    out=Path(a.output)
    if out.exists():
        p.error(f"Output exists: {out}")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
