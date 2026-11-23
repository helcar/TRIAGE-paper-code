"""Complete ViT B7 six-method tables with an exact VAL replay gate.

Run only after ViT layer/config are frozen. CAL reference fitting, FIT topic
fitting, same selected layer across all methods, and no TEST tuning.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from .triage import TRIAGE,Config
from .run_experiment import read,pred
from .validate_triage import prototype_matrices
from .evaluation import (evaluate,error_labels,confidence_score,
                         FeatureBaselines,fit_prototype_profiles,
                         prototype_attribution_divergence)


def pool(d,step=32):
    return np.vstack([
        np.asarray(d["features"][i:i+step]).mean(axis=1)
        for i in range(0,len(d["labels"]),step)
    ])


def summarize(data,scores):
    errors=error_labels(data["labels"],pred(data))
    confidence=1-confidence_score(data["logits"])
    groups={"overall":np.ones(len(errors),dtype=bool),
            "low_confidence":confidence<.70,
            "high_confidence":confidence>=.70}
    result={}
    for name,mask in groups.items():
        subset=errors[mask]
        result[name]={"frames":int(mask.sum()),"errors":int(subset.sum()),
                      "error_prevalence":float(subset.mean()) if len(subset) else None,
                      "metrics":{method:evaluate(subset,score[mask])
                                 for method,score in scores.items()} if len(subset) else {}}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("fit","cal","val","test","selection-json","output"):
        p.add_argument("--"+arg,required=True)
    p.add_argument("--dataset",default="FF++ c23",choices=["FF++ c23","Celeb-DF-v2","DFDC"])
    p.add_argument("--layer-ablation-json",required=True)
    p.add_argument("--layer",required=True,type=int)
    p.add_argument("--authorize-test",action="store_true")
    a=p.parse_args()
    if not a.authorize_test:
        p.error("--authorize-test required after validation selection")
    out=Path(a.output)
    if out.exists():
        p.error("Output exists; choose a new report filename")
    layer_report=json.loads(Path(a.layer_ablation_json).read_text(encoding="utf-8"))
    if a.layer!=int(layer_report["selected_layer"]):
        p.error("Layer differs from preselected validation-screen winner")
    for split,path in (("fit",a.fit),("cal",a.cal),("val",a.val),("test",a.test)):
        part=Path(path)
        if part.name!=split or f"layer{a.layer}_arrays" not in str(part.parent):
            p.error(f"Wrong {split} array directory for selected layer {a.layer}: {part}")
        marker=part/"_complete.json"
        if not marker.is_file():
            p.error(f"Incomplete {split} extraction: {part}")
        meta=json.loads(marker.read_text(encoding="utf-8"))
        if meta.get("layer")!=f"classifier.vit.layers.{a.layer}" or not meta.get("finetuned_checkpoint"):
            p.error(f"Unexpected ViT checkpoint/layer provenance for {split}")
    selected=json.loads(Path(a.selection_json).read_text(encoding="utf-8"))
    best=selected["best"]["full_topic"]
    search=selected["search"]
    if search["max_local_features"]!=50000 or search["seed"]!=42:
        p.error("Unexpected search provenance")
    cfg=Config(prototypes=best["K"],topics_positive=best["M"],
               topics_negative=best["M"],references_real=best["J"],
               references_fake=best["J"],tau=best["tau"],
               lambda_support=best["lambda_support"],
               lambda_oppose=best["lambda_oppose"],
               lambda_magnitude=best["lambda_magnitude"],
               max_local_features=50000,seed=42)
    fit,cal,val=(read(path) for path in (a.fit,a.cal,a.val))
    model=TRIAGE(cfg)
    model.fit_prototypes(fit["features"][i:i+32]
                         for i in range(0,len(fit["labels"]),32))
    fproto=prototype_matrices(model,fit,32)
    cproto=prototype_matrices(model,cal,32)
    vproto=prototype_matrices(model,val,32)
    model.fit_topics(*fproto)
    model.fit_references(model.encode(*cproto),cal["labels"],pred(cal))
    vs=model.score(model.encode(*vproto),pred(val))
    ve=error_labels(val["labels"],pred(val))
    auc=float(roc_auc_score(ve,vs))
    ap=float(average_precision_score(ve,vs))
    if abs(auc-best["auroc"])>1e-7 or abs(ap-best["auprc"])>1e-7:
        raise RuntimeError("Validation TRIAGE replay mismatch. TEST not opened")
    print("VAL replay verified:",auc,ap,flush=True)

    # CAL-only fitting for all non-TRIAGE comparison models.
    feature_baselines=FeatureBaselines().fit(
        pool(cal),cal["labels"],pred(cal))
    pad_profiles=fit_prototype_profiles(
        *cproto,cal["labels"],pred(cal))
    test=read(a.test)
    tproto=prototype_matrices(model,test,32)
    td=feature_baselines.score(pool(test),pred(test))
    scores={"Confidence":confidence_score(test["logits"]),
            "Centroid":td["centroid"],
            "Mahalanobis":td["mahalanobis"],
            "kNN":td["knn"],
            "PAD":prototype_attribution_divergence(
                *tproto,pred(test),pad_profiles),
            "TRIAGE":model.score(model.encode(*tproto),pred(test))}
    report={"source":f"ViT {a.dataset} TEST, frozen layer {a.layer}",
            "dataset":a.dataset,
            "layer":a.layer,"selection":best,
            "validation_replay":{"auroc":auc,"auprc":ap},
            "test":summarize(test,scores)}
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    for subset in ("overall","low_confidence","high_confidence"):
        entry=report["test"][subset]
        print(f"\n{subset}: frames={entry['frames']} errors={entry['errors']}")
        for name,v in entry["metrics"].items():
            print(f"{name:13s} AUROC={v['auroc']} AUPRC={v['auprc']} "
                  f"AURC={v['aurc']:.4f} Review10={v['error_recall_at_budget_0.1']}")


if __name__=="__main__":
    main()
