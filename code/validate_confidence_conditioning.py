"""Group-held-out development evaluation of confidence-conditioned TRIAGE.

Uses fit/cal/val arrays only. Splits official validation *source groups*
into development and assessment (never individual frames); fits score
combination exclusively on development. Assessment is diagnostic, NOT the
official test result. Reuses saved Xception features and attributions.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .triage import TRIAGE, Config
from .run_experiment import read, pred
from .evaluation import confidence_score, error_labels
from .validate_triage import prototype_matrices


def summary(y, s):
    result={"n":int(len(y)),"errors":int(y.sum()),"prevalence":float(y.mean()) if len(y) else None}
    if len(np.unique(y)) == 2:
        result.update(auroc=float(roc_auc_score(y,s)),
                      auprc=float(average_precision_score(y,s)))
    else:
        result.update(auroc=None,auprc=None)
    return result


def load_validation_groups(manifest, n, expected_labels):
    with open(manifest,newline="",encoding="utf-8-sig") as f:
        rows=[r for r in csv.DictReader(f) if r["split"]=="val"]
    if len(rows)!=n:
        raise ValueError(f"Validation manifest has {len(rows)} rows, arrays have {n}; use the EXACT extraction manifest")
    y=np.array([int(r["label"]) for r in rows])
    if not np.array_equal(y,expected_labels):
        raise ValueError("Manifest and extracted validation labels differ; group alignment cannot be verified")
    groups=np.array([r["group_id"] for r in rows],dtype=str)
    if any(not g for g in groups): raise ValueError("Missing source group ID")
    return groups,rows


def build_features(conf, triage, magnitude, gate):
    # Fixed threshold 0.7; logistic fit estimates the significance and direction
    # of signal, while learned coefficients use development groups only.
    low=(conf>0.3).astype(float)
    if gate:
        return np.column_stack((conf,triage,magnitude,low,triage*low,magnitude*low))
    return np.column_stack((conf,triage))


def run(args):
    fit,cal,val=(read(p) for p in (args.fit,args.cal,args.val))
    groups,rows=load_validation_groups(args.manifest,len(val["labels"]),np.asarray(val["labels"]))
    cfg=Config(prototypes=args.prototypes,topics_positive=args.topics,
               topics_negative=args.topics,references_real=args.references,
               references_fake=args.references,max_local_features=args.max_local_features,
               seed=args.seed)
    model=TRIAGE(cfg)
    batch=args.transform_batch
    model.fit_prototypes(fit["features"][i:i+batch] for i in range(0,len(fit["labels"]),batch))
    proto_fit=prototype_matrices(model,fit,batch)
    proto_cal=prototype_matrices(model,cal,batch)
    proto_val=prototype_matrices(model,val,batch)
    model.fit_topics(*proto_fit)
    model.fit_references(model.encode(*proto_cal),cal["labels"],pred(cal))
    enc=model.encode(*proto_val)
    triage=model.score(enc,pred(val))
    confidence=confidence_score(val["logits"])
    magnitude=sum(enc[q]["a"] for q in ("+","-"))
    errors=error_labels(val["labels"],pred(val))
    splitter=GroupShuffleSplit(n_splits=1,test_size=args.assessment_fraction,random_state=args.seed)
    dev,assessment=next(splitter.split(confidence,errors,groups))
    if np.unique(errors[dev]).size!=2 or np.unique(errors[assessment]).size!=2:
        raise ValueError("Development/assessment must each contain errors and correct samples; use a different prespecified seed")
    if set(groups[dev]) & set(groups[assessment]):
        raise AssertionError("Group leakage")
    def learn(x):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced"))
        clf.fit(x[dev],errors[dev])
        return clf.predict_proba(x[assessment])[:,1]
    basic=learn(build_features(confidence,triage,magnitude,False))
    conditioned=learn(build_features(confidence,triage,magnitude,True))
    outcome={"protocol":{"seed":args.seed,"low_confidence_threshold":0.7,
                          "source_groups_development":len(set(groups[dev])),
                          "source_groups_assessment":len(set(groups[assessment])),
                          "samples_development":len(dev),"errors_development":int(errors[dev].sum()),
                          "samples_assessment":len(assessment),"errors_assessment":int(errors[assessment].sum())},
             "assessment":{},"assessment_low_confidence":{}}
    scores={"Confidence":confidence[assessment],"TRIAGE":triage[assessment],
            "Confidence_plus_TRIAGE":basic,
            "Confidence_conditioned_TRIAGE":conditioned}
    mask=confidence[assessment]>0.3
    for name,score in scores.items():
        outcome["assessment"][name]=summary(errors[assessment],score)
        outcome["assessment_low_confidence"][name]=summary(errors[assessment][mask],score[mask])
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(outcome,indent=2),encoding="utf-8")
    if args.scores_csv:
        csv_path=Path(args.scores_csv);csv_path.parent.mkdir(parents=True,exist_ok=True)
        with csv_path.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["group_id","error","confidence_risk","TRIAGE",
                                        "Confidence_plus_TRIAGE","Confidence_conditioned_TRIAGE"])
            for i,j in enumerate(assessment):
                w.writerow([groups[j],int(errors[j]),float(confidence[j]),float(triage[j]),
                            float(basic[i]),float(conditioned[i])])
    return outcome


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("fit","cal","val","manifest","output"):p.add_argument("--"+k,required=True)
    p.add_argument("--scores-csv")
    p.add_argument("--prototypes",type=int,default=128)
    p.add_argument("--topics",type=int,default=16)
    p.add_argument("--references",type=int,default=2)
    p.add_argument("--max-local-features",type=int,default=50000)
    p.add_argument("--transform-batch",type=int,default=64)
    p.add_argument("--assessment-fraction",type=float,default=.5)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if not 0<args.assessment_fraction<1:p.error("assessment-fraction must be in (0,1)")
    print(json.dumps(run(args),indent=2))


if __name__=="__main__":main()
