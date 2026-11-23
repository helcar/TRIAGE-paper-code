"""Reproduce the THREE Xception-style error-detection tables for another backbone.

Per dataset: read FIT/CAL/VAL; load the preselected *full-topic* setting from
the saved validation grid; refit TRIAGE on FIT, references on CAL; reproduce
VAL AUROC/AUPRC exactly; then, only with --evaluate-test, access TEST and
compute the six original baselines for ALL / confidence >=70% / <70%.

TEST is never used for tuning. Feature-based baselines and PAD use CAL
correct-prediction references only. Outputs are descriptive test metrics
and an aligned per-frame CSV for optional group-aware inference.

Command example is in code/EFFICIENTNET_B4_STUDY.md.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluation import (
    FeatureBaselines, confidence_score, error_labels, evaluate,
    fit_prototype_profiles, prototype_attribution_divergence,
)
from .run_experiment import read, pred, pooled_features
from .triage import Config, TRIAGE
from .validate_triage import prototype_matrices

METHODS=(
    "Prediction Confidence",
    "Feature Centroid Distance",
    "Mahalanobis Distance",
    "kNN Feature Distance",
    "Prototype Attribution Divergence",
    "TRIAGE",
)
SPLITS=("fit","cal","val","test")
SCOPE_NAMES=(
    "overall",
    "high_confidence_ge_70pct",
    "low_confidence_lt_70pct",
)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def manifest_rows(path):
    with Path(path).open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError("Frame manifest is empty")
    for column in ("path","split","label","group_id","video_id"):
        if column not in rows[0]:
            raise ValueError(f"Manifest missing {column}")
    assignments={}
    for r in rows:
        if r["split"] not in SPLITS or r["label"] not in ("0","1"):
            raise ValueError("Unexpected split or label in manifest")
        group=r["group_id"]
        if not group:
            raise ValueError("Empty group_id")
        old=assignments.setdefault(group,r["split"])
        if old!=r["split"]:
            raise ValueError(f"Manifest group {group} shared by {old} and {r['split']}")
    return {k:[r for r in rows if r["split"]==k] for k in SPLITS}


def inspect_split(root,key,expected_manifest_sha,prior=None):
    folder=Path(root)/key
    with (folder/"_complete.json").open(encoding="utf-8") as f:
        meta=json.load(f)
    if meta.get("detector")!="DeepfakeBench EfficientNetB4":
        raise ValueError(f"{folder}: wrong detector identity: {meta.get('detector')}")
    if meta.get("manifest_sha256")!=expected_manifest_sha:
        raise ValueError(f"{folder}: differs from supplied manifest SHA256")
    provenance={k:meta.get(k) for k in (
        "detector","checkpoint_sha256","manifest_sha256","layer",
        "image_size","normalization","loaded_tensors"
    )}
    if not provenance["checkpoint_sha256"]:
        raise ValueError(f"{folder}: checkpoint identity missing")
    if prior is not None and provenance!=prior:
        raise ValueError(f"{folder}: checkpoint, layer or preprocessing changed")
    return provenance


def aligned(data,rows,split):
    if len(rows)!=len(data["labels"]):
        raise ValueError(f"{split}: labels and manifest have different lengths")
    if not np.array_equal(
            np.array([int(r["label"]) for r in rows],dtype=int),
            np.asarray(data["labels"]).reshape(-1)):
        raise ValueError(f"{split}: label sequence differs from manifest order")
    if not np.isfinite(data["logits"]).all():
        raise ValueError(f"{split}: non-finite detector logits")
    return True


def selected_config(gridfile):
    report=json.loads(Path(gridfile).read_text(encoding="utf-8"))
    grid=report["search"]
    best=report["best"]["full_topic"]
    if best.get("kind")!="full_topic":
        raise ValueError("Only preselected full-topic TRIAGE allowed here")
    seed=int(grid["seed"])
    k,m,j=int(best["K"]),int(best["M"]),int(best["J"])
    cfg=Config(
        prototypes=k,topics_positive=m,topics_negative=m,
        references_real=j,references_fake=j,
        tau=float(best["tau"]),
        lambda_support=float(best["lambda_support"]),
        lambda_oppose=float(best["lambda_oppose"]),
        lambda_magnitude=float(best["lambda_magnitude"]),
        max_local_features=int(grid["max_local_features"]),
        seed=seed,
    )
    return cfg,best,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True)
    p.add_argument("--arrays-root",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--grid-json",required=True)
    p.add_argument("--output-json",required=True)
    p.add_argument("--scores-csv")
    p.add_argument("--evaluate-test",action="store_true")
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--tolerance",type=float,default=1e-5)
    a=p.parse_args()
    if a.batch<=0 or a.tolerance<=0:
        p.error("Positive batch/tolerance required")
    if a.evaluate_test and not a.scores_csv:
        p.error("--scores-csv is required when evaluating TEST")

    manifest=manifest_rows(a.manifest)
    original_sha=digest(a.manifest)
    cfg,best,grid=selected_config(a.grid_json)
    provenance=None
    arrays={}
    # TEST does not get opened, fingerprinted or inspected until after
    # the selected validation result is reproduced.
    for key in ("fit","cal","val"):
        provenance=inspect_split(a.arrays_root,key,original_sha,provenance)
        arrays[key]=read(Path(a.arrays_root)/key)
        aligned(arrays[key],manifest[key],key)

    fit,cal,val=arrays["fit"],arrays["cal"],arrays["val"]
    model=TRIAGE(cfg)
    b=a.batch
    model.fit_prototypes(
        fit["features"][i:i+b] for i in range(0,len(fit["labels"]),b)
    )
    fp=prototype_matrices(model,fit,b)
    cp=prototype_matrices(model,cal,b)
    model.fit_topics(*fp)
    model.fit_references(model.encode(*cp),cal["labels"],pred(cal))
    vp=prototype_matrices(model,val,b)
    val_errors=error_labels(val["labels"],pred(val))
    val_score=model.score(model.encode(*vp),pred(val))
    val_metrics=evaluate(val_errors,val_score)
    conf_metrics=evaluate(val_errors,confidence_score(val["logits"]))
    for name,observed,expected in (
        ("TRIAGE AUROC",val_metrics["auroc"],best["auroc"]),
        ("TRIAGE AUPRC",val_metrics["auprc"],best["auprc"]),
        ("confidence AUROC",conf_metrics["auroc"],grid["confidence"]["AUROC"]),
        ("confidence AUPRC",conf_metrics["auprc"],grid["confidence"]["AUPRC"]),
    ):
        if observed is None or abs(observed-expected)>a.tolerance:
            raise RuntimeError(
                f"Validation reproduction FAILED ({name}): "
                f"{observed} vs {expected}. TEST has not been accessed.")
    print(f"VAL reproduced: n={len(val_errors)}, errors={int(val_errors.sum())}, "
          f"TRIAGE AUROC={val_metrics['auroc']:.9f}, "
          f"AUPRC={val_metrics['auprc']:.9f}",flush=True)

    report={
        "dataset":a.dataset,
        "detector":"DeepfakeBench EfficientNetB4",
        "layer":provenance["layer"],
        "provenance":provenance,
        "configuration":{
            "K":cfg.prototypes,"M":cfg.topics_positive,
            "J":cfg.references_real,"tau":cfg.tau,
            "lambda_support":cfg.lambda_support,
            "lambda_oppose":cfg.lambda_oppose,
            "lambda_magnitude":cfg.lambda_magnitude,
            "seed":cfg.seed,"max_local_features":cfg.max_local_features,
        },
        "validation_selection":{"triage":val_metrics,
                                "confidence":conf_metrics,
                                "same_data_used_for_selection":True},
        "test_evaluated":False,
        "methods":list(METHODS),
        "confidence_cutoff":.70,
    }
    if a.evaluate_test:
        inspect_split(a.arrays_root,"test",original_sha,provenance)
        test=read(Path(a.arrays_root)/"test")
        aligned(test,manifest["test"],"test")
        tp=prototype_matrices(model,test,b)
        risk=model.score(model.encode(*tp),pred(test))
        uncertainty=confidence_score(test["logits"])
        features=FeatureBaselines().fit(
            pooled_features(cal),cal["labels"],pred(cal))
        dist=features.score(pooled_features(test),pred(test))
        pad_refs=fit_prototype_profiles(
            *cp,cal["labels"],pred(cal))
        pad=prototype_attribution_divergence(
            *tp,pred(test),pad_refs)
        scores={
            "Prediction Confidence":uncertainty,
            "Feature Centroid Distance":dist["centroid"],
            "Mahalanobis Distance":dist["mahalanobis"],
            "kNN Feature Distance":dist["knn"],
            "Prototype Attribution Divergence":pad,
            "TRIAGE":risk,
        }
        y=error_labels(test["labels"],pred(test))
        masks={
            "overall":np.ones(len(y),dtype=bool),
            "high_confidence_ge_70pct":uncertainty<=.30,
            "low_confidence_lt_70pct":uncertainty>.30,
        }
        partitions={}
        for scope,mask in masks.items():
            sub=y[mask]
            partitions[scope]={
                "n":int(mask.sum()),
                "errors":int(sub.sum()),
                "source_groups":len({
                    manifest["test"][i]["group_id"]
                    for i in np.flatnonzero(mask)
                }),
                "methods":{
                    name:(evaluate(sub,score[mask])
                          if len(sub) else None)
                    for name,score in scores.items()
                },
            }
        if (partitions["high_confidence_ge_70pct"]["n"]+
            partitions["low_confidence_lt_70pct"]["n"]!=len(y)):
            raise RuntimeError("Confidence strata do not partition TEST")
        dest=Path(a.scores_csv)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f)
            writer.writerow(
                ["group_id","video_id","method","true_label",
                 "predicted_label","error","confidence_group"]
                +list(METHODS))
            for i,r in enumerate(manifest["test"]):
                writer.writerow([
                    r["group_id"],r["video_id"],r.get("method",""),
                    int(test["labels"][i]),int(pred(test)[i]),
                    int(y[i]),"low" if uncertainty[i]>.30 else "high",
                ]+[float(scores[method][i]) for method in METHODS])
        report["test_evaluated"]=True
        report["test"]={
            "n":len(y),"errors":int(y.sum()),
            "class_counts":{
                "real":int((np.asarray(test["labels"])==0).sum()),
                "fake":int((np.asarray(test["labels"])==1).sum()),
            },
            "sections":partitions,
            "scores_csv":str(dest),
        }
        for scope,obj in partitions.items():
            print(f"\n{scope}: n={obj['n']} errors={obj['errors']}",flush=True)
            for name,v in obj["methods"].items():
                if v is None:
                    print(f"  {name}: AUROC=n/a, AUPRC=n/a")
                else:
                    print(f"  {name}: AUROC={v['auroc']}, AUPRC={v['auprc']}")
    output=Path(a.output_json)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"\nSaved {output}",flush=True)


if __name__=="__main__":
    main()
