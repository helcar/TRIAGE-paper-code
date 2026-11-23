"""Staged validation-only TRIAGE sensitivity search with audit CSV.

Uses existing fit/cal/val arrays only, never reads test. Efficiently reuses
prototype assignments, NMF models, encodings and reference profiles. A
strictly positive topic-weight branch is tracked separately from a
magnitude-only ablation; M and tau cannot affect magnitude-only scores.
"""
import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .triage import TRIAGE, Config, js_components
from .run_experiment import read, pred
from .evaluation import confidence_score, error_labels
from .validate_triage import prototype_matrices


def metrics(y, score):
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def raw_reference_terms(model, enc, predictions):
    """Compute [N,J] reference-specific support, oppose, magnitude terms.

    Preserve reference selection as min of the *weighted total*; independently
    minimizing each term would not reproduce the TRIAGE method.
    """
    n=len(predictions)
    output={}
    for c in (0,1):
        indices=np.flatnonzero(predictions==c)
        sup,opp=("+","-") if c==1 else ("-","+")
        j=len(model.profiles[c])
        terms=np.empty((len(indices),j,3),dtype=float)
        for t,i in enumerate(indices):
            for k,ref in enumerate(model.profiles[c]):
                terms[t,k,0]=enc[sup]["eta"][i]*js_components(enc[sup]["p"][i],ref[sup]).sum()
                terms[t,k,1]=enc[opp]["eta"][i]*js_components(enc[opp]["p"][i],ref[opp]).sum()
                terms[t,k,2]=sum(abs(np.log1p(enc[q]["a"][i])-ref["mu"+q]) for q in ("+","-"))
        output[c]=(indices,terms)
    return output


def score_terms(terms_by_class, size, weights):
    score=np.empty(size,dtype=float)
    w=np.asarray(weights,dtype=float)
    for indices,terms in terms_by_class.values():
        score[indices]=np.min(terms@w,axis=1)
    return score


def search(fit,cal,val,k_values=(64,128,256),m_values=(4,8,16),
           j_values=(1,2,4),max_local_features=50000,batch=64,seed=42):
    arrays={q:read(p) for q,p in (("fit",fit),("cal",cal),("val",val))}
    y=error_labels(arrays["val"]["labels"],pred(arrays["val"]))
    if len(np.unique(y))!=2:raise ValueError("Validation needs both outcomes")
    # Restrict computationally expensive search: 3 K x 3 M x 3 J;
    # then reuse NMF to explore score weights and tau.
    topic_weights=[
        (1.,1.,0.),(1.,1.,.1),(1.,1.,1.),
        (1.,0.,.1),(0.,1.,.1),(.1,.1,1.),
        (.01,.01,1.),(1.,.1,.1),(.1,1.,.1),
        (2.,1.,.1),(1.,2.,.1),
    ]
    tau_values=(.01,.1,1.,10.,100.)
    records=[]
    for k in k_values:
        base=TRIAGE(Config(prototypes=k,topics_positive=1,topics_negative=1,
                           max_local_features=max_local_features,seed=seed))
        a=arrays["fit"]
        base.fit_prototypes(a["features"][i:i+batch] for i in range(0,len(a["labels"]),batch))
        proto={name:prototype_matrices(base,data,batch) for name,data in arrays.items()}
        for m in m_values:
            if m>k:continue
            cfg=replace(base.cfg,topics_positive=m,topics_negative=m)
            model=TRIAGE(cfg);model.prototypes=base.prototypes
            model.fit_topics(*proto["fit"])
            enc_cal=model.encode(*proto["cal"])
            for j in j_values:
                model.cfg=replace(model.cfg,references_real=j,references_fake=j)
                model.fit_references(enc_cal,arrays["cal"]["labels"],pred(arrays["cal"]))
                for tau in tau_values:
                    model.cfg=replace(model.cfg,tau=tau)
                    enc_val=model.encode(*proto["val"])
                    terms=raw_reference_terms(model,enc_val,pred(arrays["val"]))
                    for w in topic_weights:
                        score=score_terms(terms,len(y),w)
                        auc,ap=metrics(y,score)
                        records.append(dict(kind="full_topic",K=k,M=m,J=j,tau=tau,
                                            lambda_support=w[0],lambda_oppose=w[1],
                                            lambda_magnitude=w[2],auroc=auc,auprc=ap))
                # magnitude-only is invariant to M and tau. Compute once per K,J
                if m==min(v for v in m_values if v<=k):
                    enc_val=model.encode(*proto["val"])
                    terms=raw_reference_terms(model,enc_val,pred(arrays["val"]))
                    score=score_terms(terms,len(y),(0.,0.,1.))
                    auc,ap=metrics(y,score)
                    records.append(dict(kind="magnitude_only",K=k,M=None,J=j,tau=None,
                                        lambda_support=0.,lambda_oppose=0.,
                                        lambda_magnitude=1.,auroc=auc,auprc=ap))
            print(f"Completed K={k}, M={m}: {len(records)} candidate rows",flush=True)
    best={}
    for kind in ("full_topic","magnitude_only"):
        candidates=[r for r in records if r["kind"]==kind]
        best[kind]=max(candidates,key=lambda r:(r["auroc"],r["auprc"]))
    return {"validation_images":len(y),"validation_errors":int(y.sum()),
            "search":{"K":list(k_values),"M":list(m_values),"J":list(j_values),
                      "tau":list(tau_values),"topic_weight_sets":[list(w) for w in topic_weights],
                      "max_local_features":max_local_features,"seed":seed},
            "candidates_evaluated":len(records),
            "best":best,
            "confidence":{"AUROC":metrics(y,confidence_score(arrays["val"]["logits"]))[0],
                          "AUPRC":metrics(y,confidence_score(arrays["val"]["logits"]))[1]},
            "warning":"Model selection on validation: these are NOT untouched test results."},records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("fit","cal","val","output","candidates-csv"):p.add_argument("--"+name,required=True)
    p.add_argument("--k",type=int,nargs="+",default=[64,128,256])
    p.add_argument("--m",type=int,nargs="+",default=[4,8,16])
    p.add_argument("--j",type=int,nargs="+",default=[1,2,4])
    p.add_argument("--max-local-features",type=int,default=50000)
    p.add_argument("--transform-batch",type=int,default=64)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if min(*args.k,*args.m,*args.j,args.max_local_features,args.transform_batch)<1:
        p.error("Count parameters must be positive")
    report,records=search(args.fit,args.cal,args.val,tuple(args.k),tuple(args.m),
                          tuple(args.j),args.max_local_features,args.transform_batch,args.seed)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    csvpath=Path(args.candidates_csv);csvpath.parent.mkdir(parents=True,exist_ok=True)
    with csvpath.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
