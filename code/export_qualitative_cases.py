"""Generate auditable TRIAGE example panels from cached block8 arrays.

Fits only on FIT, builds references on CAL, reproduces VAL and TEST AUROC,
then selects the example closest to median TRIAGE risk in each detector
outcome group: TP, TN, FP and FN. This deterministic, descriptive selection
must not be interpreted as representative of the score distribution at large.

Each panel shows an actual cropped test frame, its signed grad*activation
local evidence (red=fake support, blue=real support), and the active-polarity
NMF topic profile compared with the selected correctly-classified reference.
The figure is generated from aligned existing features/attributions, without
rerunning Xception or changing detector weights.

Also exports full metadata and a separate list of high-confidence detector
errors recovered by TRIAGE at the GLOBAL top-10% review budget.
"""
import argparse
import csv
import json
import math
import warnings
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning

from .evaluation import error_labels, evaluate
from .run_experiment import read, pred
from .triage import Config, TRIAGE
from .validate_triage import prototype_matrices


def load_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    needed={"split","label","path"}
    if not rows or not needed.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(needed)}")
    return {k:[r for r in rows if r["split"]==k]
            for k in ("fit","cal","val","test")}


def check_alignment(manifest, splits):
    for split, data in splits.items():
        rows=manifest[split]
        if len(rows)!=len(data["labels"]):
            raise ValueError(f"{split}: manifest {len(rows)} rows but arrays "
                             f"{len(data['labels'])} rows")
        labels=np.fromiter((int(r["label"]) for r in rows),dtype=int)
        if not np.array_equal(labels,np.asarray(data["labels"])):
            raise ValueError(f"{split}: row-aligned labels disagree")


def image_path(row, image_root):
    path=Path(row["path"])
    return path if path.is_absolute() else Path(image_root)/path


def signed_grid(attributions):
    attr=np.asarray(attributions,dtype=float)
    t=attr.size
    side=math.isqrt(t)
    if side*side!=t:
        raise ValueError(f"Local token count {t} is not square; "
                         "cannot recover a spatial grid without explicit geometry")
    return attr.reshape(side,side)


def serialize_example(i, category, rows, data, predictions, confidence, score,
                      explanations, encoded, model, top10):
    c=int(predictions[i])
    polarity=("-" if c==1 else "+") if model.cfg.lambda_oppose>model.cfg.lambda_support else (
        "+" if c==1 else "-")
    explanation=explanations[i]
    ref=model.profiles[c][int(explanation["reference_index"])]
    p=np.asarray(encoded[polarity]["p"][i],dtype=float)
    q=np.asarray(ref[polarity],dtype=float)
    topic_contrib=np.asarray(
        explanation["oppose_topic_contributions"]
        if polarity==("-" if c==1 else "+")
        else explanation["support_topic_contributions"],dtype=float
    )
    order=np.argsort(-topic_contrib,kind="stable")
    # If the selected score has a zero-weight topic term, list the largest
    # distribution differences as descriptive evidence, not score causes.
    if not np.any(topic_contrib>0):
        order=np.argsort(-np.abs(p-q),kind="stable")
    top=[]
    for topic in order[:min(3,len(order))]:
        comp=model.nmf[polarity].components_[int(topic)]
        proto_order=np.argsort(-comp,kind="stable")[:min(5,len(comp))]
        cols=model.columns[polarity]
        top.append({
            "topic":int(topic),
            "active_score_contribution":float(topic_contrib[topic]),
            "sample_probability":float(p[topic]),
            "matched_reference_probability":float(q[topic]),
            "delta":float(p[topic]-q[topic]),
            "highest_weight_prototypes":[
                {"prototype":int(cols[k]),"nmf_component_weight":float(comp[k])}
                for k in proto_order
            ],
        })
    return {
        "test_index":int(i),"outcome":category,
        "path":rows[i]["path"],
        "group_id":rows[i].get("group_id"),
        "video_id":rows[i].get("video_id"),
        "true_label":int(data["labels"][i]),
        "predicted_label":c,
        "confidence":float(confidence[i]),
        "triage_risk_score":float(score[i]),
        "in_global_top10pct":bool(i in top10),
        "reference_index":int(explanation["reference_index"]),
        "plotted_polarity":polarity,
        "topic_distribution":p.tolist(),
        "matched_reference_distribution":q.tolist(),
        "magnitude_contribution":float(explanation["magnitude_contribution"]),
        "top_topics":top,
    }


def choose_median_candidates(labels,predictions,score):
    groups={
        "TP":(labels==1)&(predictions==1),
        "TN":(labels==0)&(predictions==0),
        "FP":(labels==0)&(predictions==1),
        "FN":(labels==1)&(predictions==0),
    }
    out={}
    for name,mask in groups.items():
        ids=np.flatnonzero(mask)
        if not ids.size:
            raise ValueError(f"Test partition has no examples for {name}")
        median=np.median(score[ids])
        out[name]=int(ids[np.argmin(np.abs(score[ids]-median))])
    return out


def save_figure(chosen,report,manifest,data,model,encoded,images_root,
                output_pdf,output_png,title):
    fig, axes=plt.subplots(4,3,figsize=(12.8,12.0),
                          gridspec_kw={"width_ratios":[1,1,1.6]})
    all_abs=np.concatenate([
        np.abs(np.asarray(data["attributions"][i],float)).ravel()
        for i in chosen.values()
    ])
    max_abs=max(float(np.quantile(all_abs,.98)),1e-12)
    for row_num,(name,i) in enumerate(chosen.items()):
        entry=report["examples"][name]
        path=image_path(manifest["test"][i],images_root)
        if not path.is_file():
            raise FileNotFoundError(f"Face crop not found: {path}. "
                                    "Check --image-root and manifest path.")
        with Image.open(path) as image:
            face=np.asarray(image.convert("RGB").resize((256,256)))
        attr=signed_grid(data["attributions"][i])
        ax0,ax1,ax2=axes[row_num]
        ax0.imshow(face)
        ax0.set_title(f"{name} / {['REAL','FAKE'][entry['predicted_label']]} "
                      f"pred, conf={entry['confidence']:.3f}",fontsize=9)
        ax1.imshow(face)
        ax1.imshow(attr,extent=(0,256,256,0),
                   cmap="coolwarm",vmin=-max_abs,vmax=max_abs,
                   alpha=.55,interpolation="nearest")
        ax1.set_title(f"Signed local evidence (red=FAKE)\n"
                      f"TRIAGE risk={entry['triage_risk_score']:.3f}",
                      fontsize=9)
        p=np.asarray(entry["topic_distribution"])
        q=np.asarray(entry["matched_reference_distribution"])
        x=np.arange(len(p))
        width=.40
        ax2.bar(x-width/2,p,width,label="Current prediction")
        ax2.bar(x+width/2,q,width,label="Correct-reference profile")
        ax2.set_ylim(0,max(.25,1.12*max(float(p.max()),float(q.max()))))
        ax2.set_xlim(-.75,len(p)-.25)
        ax2.set_ylabel("Topic probability",fontsize=8)
        ax2.set_xlabel(f"Topic index ({'fake' if entry['plotted_polarity']=='+' else 'real'}-supporting polarity)",fontsize=8)
        ax2.set_title(f"Reference #{entry['reference_index']} / "
                      f"global top 10% risk: {entry['in_global_top10pct']}",
                      fontsize=9)
        ax2.set_xticks(x if len(x)<=16 else x[::max(1,len(x)//12)])
        ax2.tick_params(labelsize=7)
        if row_num==0:ax2.legend(fontsize=7,loc="upper right")
        for ax in (ax0,ax1):ax.axis("off")
    fig.suptitle(title+" | Median-risk example within each outcome group",
                 fontsize=12)
    fig.text(.5,.012,
             "Image crops and signed attributions are recorded data; "
             "examples are descriptive, not error-rate estimates.",
             ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.96))
    output_pdf.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_pdf,bbox_inches="tight")
    fig.savefig(output_png,dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("fit","cal","val","test","manifest","output-pdf",
                "output-png","output-json"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--image-root",default=".")
    p.add_argument("--dataset-name",default="FF++")
    p.add_argument("--prototypes",type=int,required=True)
    p.add_argument("--topics",type=int,required=True)
    p.add_argument("--references",type=int,required=True)
    p.add_argument("--tau",type=float,required=True)
    p.add_argument("--lambda-support",type=float,required=True)
    p.add_argument("--lambda-oppose",type=float,required=True)
    p.add_argument("--lambda-magnitude",type=float,required=True)
    p.add_argument("--expected-val-auroc",type=float,required=True)
    p.add_argument("--expected-test-auroc",type=float,required=True)
    p.add_argument("--max-local-features",type=int,default=50000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--tolerance",type=float,default=1e-5)
    a=p.parse_args()
    arrays={key:read(getattr(a,key)) for key in ("fit","cal","val","test")}
    manifest=load_manifest(a.manifest)
    check_alignment(manifest,arrays)
    cfg=Config(prototypes=a.prototypes,topics_positive=a.topics,
               topics_negative=a.topics,references_real=a.references,
               references_fake=a.references,tau=a.tau,
               lambda_support=a.lambda_support,
               lambda_oppose=a.lambda_oppose,
               lambda_magnitude=a.lambda_magnitude,
               max_local_features=a.max_local_features,seed=a.seed)
    model=TRIAGE(cfg)
    fit,cal,val,test=(arrays[k] for k in ("fit","cal","val","test"))
    model.fit_prototypes(
        fit["features"][i:i+a.batch] for i in range(0,len(fit["labels"]),a.batch)
    )
    fp=prototype_matrices(model,fit,a.batch)
    cp=prototype_matrices(model,cal,a.batch)
    vp=prototype_matrices(model,val,a.batch)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always",ConvergenceWarning)
        model.fit_topics(*fp)
    model.fit_references(model.encode(*cp),cal["labels"],pred(cal))
    val_score=model.score(model.encode(*vp),pred(val))
    tv=evaluate(error_labels(val["labels"],pred(val)),val_score)
    if abs(tv["auroc"]-a.expected_val_auroc)>a.tolerance:
        raise RuntimeError(f"VAL AUROC mismatch {tv['auroc']} vs "
                           f"{a.expected_val_auroc}; TEST not accessed")
    tp=prototype_matrices(model,test,a.batch)
    enc=model.encode(*tp)
    risk, explanations=model.score(enc,pred(test),explain=True)
    te=evaluate(error_labels(test["labels"],pred(test)),risk)
    if abs(te["auroc"]-a.expected_test_auroc)>a.tolerance:
        raise RuntimeError(f"TEST AUROC mismatch {te['auroc']} "
                           f"vs {a.expected_test_auroc}; figure not generated")
    logits=np.asarray(test["logits"])
    conf=np.maximum(expit(logits),1-expit(logits))
    chosen=choose_median_candidates(np.asarray(test["labels"]),pred(test),risk)
    topn=max(1,int(math.ceil(.10*len(risk))))
    top10=set(np.argsort(risk)[-topn:].tolist())
    examples={
        group:serialize_example(i,group,manifest["test"],test,pred(test),
                                conf,risk,explanations,enc,model,top10)
        for group,i in chosen.items()
    }
    mistakes=np.flatnonzero(
        error_labels(test["labels"],pred(test)) & (conf>=.90)
    )
    recovered=sorted(
        (int(i) for i in mistakes if int(i) in top10),
        key=lambda i:-risk[i]
    )
    extras=[
        serialize_example(i,"HIGH_CONF_ERROR",manifest["test"],test,pred(test),
                          conf,risk,explanations,enc,model,top10)
        for i in recovered
    ]
    output={
        "dataset":a.dataset_name,"config":asdict(cfg),
        "validation_auroc":tv["auroc"],"test_auroc":te["auroc"],
        "test_auprc":te["auprc"],"selection_rule":
        "Closest TRIAGE score to the median risk within TP/TN/FP/FN respectively",
        "global_review_budget":.1,
        "global_review_count":topn,
        "high_confidence_recovered_count":len(extras),
        "high_confidence_recovered":extras,
        "nmf_convergence_warning":[str(x.message) for x in ws
                                    if issubclass(x.category,ConvergenceWarning)],
        "examples":examples
    }
    output_json=Path(a.output_json)
    output_json.parent.mkdir(parents=True,exist_ok=True)
    output_json.write_text(json.dumps(output,indent=2),encoding="utf-8")
    save_figure(chosen,output,manifest,test,model,enc,a.image_root,
                Path(a.output_pdf),Path(a.output_png),a.dataset_name)
    print(f"VAL AUROC reproduced: {tv['auroc']:.9f}")
    print(f"TEST AUROC reproduced: {te['auroc']:.9f}")
    for group,entry in examples.items():
        print(f"{group}: test index={entry['test_index']}, "
              f"conf={entry['confidence']:.3f}, "
              f"TRIAGE score={entry['triage_risk_score']:.4f}")
    print(f"High-confidence errors recovered at GLOBAL 10% budget: {len(extras)}")
    print(f"Saved: {a.output_pdf}, {a.output_png}, {a.output_json}")


if __name__=="__main__":
    main()
