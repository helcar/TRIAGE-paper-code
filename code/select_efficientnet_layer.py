"""Choose an EfficientNet-B4 attribution layer using FIT/CAL/VAL ONLY.

Compare equally searched layers from previously generated arrays and
code.search_triage_sensitivity JSONs. Confirm detector checkpoint, source
manifest, feature layer and preprocessing provenance; exactly equal row
orders and detector decisions; identical predeclared parameter grids.

This script does not open TEST or fit any model. Run the SAME K/M/J/tau/weight
grid for each candidate layer before using it here. For FF++ after previously
viewing TEST, --post-test-exploratory is REQUIRED and the choice is not a
new independent test selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .evaluation import confidence_score, error_labels


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def load_candidate(name, root, grid_path, manifest_sha):
    root=Path(root)
    grid=json.loads(Path(grid_path).read_text(encoding="utf-8"))
    if grid.get("validation_images",0)<1:
        raise ValueError(f"{name}: invalid validation record")
    winner=grid.get("best",{}).get("full_topic",{})
    if winner.get("kind")!="full_topic":
        raise ValueError(f"{name}: no full-topic winner")
    if not all(key in winner for key in (
        "K","M","J","tau","lambda_support","lambda_oppose",
        "lambda_magnitude","auroc","auprc")):
        raise ValueError(f"{name}: incomplete selected configuration")
    provenance=None
    arrays={}
    for split in ("fit","cal","val"):
        directory=root/split
        marker=directory/"_complete.json"
        if not marker.is_file():
            raise FileNotFoundError(f"{name}: required FIT/CAL/VAL marker {marker}")
        meta=json.loads(marker.read_text(encoding="utf-8"))
        keys=("detector","checkpoint_sha256","manifest_sha256",
              "layer","image_size","normalization","loaded_tensors")
        p={key:meta.get(key) for key in keys}
        if p["detector"]!="DeepfakeBench EfficientNetB4":
            raise ValueError(f"{name}: non-EfficientNet extraction in {split}")
        if p["manifest_sha256"]!=manifest_sha:
            raise ValueError(f"{name}: manifest fingerprint mismatch")
        if provenance is None:
            provenance=p
        elif p!=provenance:
            raise ValueError(f"{name}: inconsistent extraction settings")
        data={}
        for field in ("logits","labels"):
            data[field]=np.load(directory/(field+".npy"),
                                mmap_mode="r",allow_pickle=False)
        if len(data["logits"])!=meta["samples"] or len(data["labels"])!=meta["samples"]:
            raise ValueError(f"{name}: marker/array lengths disagree for {split}")
        arrays[split]=data
    if not re.search(r"^backbone\.efficientnet\._blocks\.\d+$",
                     provenance["layer"] or ""):
        raise ValueError(f"{name}: unsupported layer name {provenance['layer']}")
    y=error_labels(arrays["val"]["labels"],
                   (arrays["val"]["logits"]>=0).astype(int))
    if len(y)!=grid["validation_images"] or int(y.sum())!=grid["validation_errors"]:
        raise ValueError(f"{name}: validation outcomes not reproduced")
    conf=confidence_score(arrays["val"]["logits"])
    reference=grid["confidence"]
    for key,metric in (("AUROC",roc_auc_score(y,conf)),
                       ("AUPRC",average_precision_score(y,conf))):
        if abs(metric-reference[key])>1e-5:
            raise ValueError(
                f"{name}: saved grid confidence {key} {reference[key]} "
                f"differs from actual VAL {metric}")
    if not np.isfinite(float(winner["auroc"])) or not np.isfinite(float(winner["auprc"])):
        raise ValueError(f"{name}: invalid grid metric")
    return {
        "name":name,"arrays_root":str(root),
        "grid_json":str(Path(grid_path)),
        "layer":provenance["layer"],
        "checkpoint_sha256":provenance["checkpoint_sha256"],
        "provenance":provenance,
        "search":grid["search"],
        "grid_candidates_evaluated":grid["candidates_evaluated"],
        "validation_images":int(grid["validation_images"]),
        "validation_errors":int(grid["validation_errors"]),
        "selected_config":winner,
    },arrays


def verify_aligned(reference,current,name,tolerance):
    for split in ("fit","cal","val"):
        a,b=reference[split],current[split]
        if not np.array_equal(a["labels"],b["labels"]):
            raise ValueError(f"{name}: labels differ on {split}")
        la=np.asarray(a["logits"],dtype=np.float64)
        lb=np.asarray(b["logits"],dtype=np.float64)
        if not np.isfinite(lb).all() or la.shape!=lb.shape:
            raise ValueError(f"{name}: logits invalid or shape differs on {split}")
        if np.any((la>=0)!=(lb>=0)):
            raise ValueError(f"{name}: detector predictions differ on {split}")
        if not np.allclose(la,lb,atol=tolerance,rtol=1e-5):
            raise ValueError(
                f"{name}: detector logits differ on {split}; "
                f"max_abs_delta={np.max(np.abs(la-lb)):.6g}. "
                "Check checkpoint, image preprocessing and determinism.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--candidate",nargs=3,action="append",required=True,
                   metavar=("NAME","ARRAYS_ROOT","GRID_JSON"))
    p.add_argument("--output",required=True)
    p.add_argument("--logit-atol",type=float,default=.005)
    p.add_argument("--post-test-exploratory",action="store_true",
                   help="Mark a post-hoc screen if this dataset's TEST was already viewed")
    a=p.parse_args()
    if a.logit_atol<0 or len(a.candidate)<2:
        p.error("At least two layers and nonnegative logit tolerance required")
    manifest_sha=sha256(a.manifest)
    candidates=[]
    baseline=None
    names=set()
    layer_names=set()
    for name,root,gridpath in a.candidate:
        if name in names:
            raise ValueError("Duplicate candidate name")
        names.add(name)
        rec,data=load_candidate(name,root,gridpath,manifest_sha)
        if rec["layer"] in layer_names:
            raise ValueError(f"Duplicate feature layer: {rec['layer']}")
        layer_names.add(rec["layer"])
        if candidates:
            prior=candidates[0]
            for field in ("checkpoint_sha256","image_size",
                          "normalization","loaded_tensors"):
                if rec["provenance"][field]!=prior["provenance"][field]:
                    raise ValueError(
                        f"{field} differs between feature layers")
            if rec["search"]!=prior["search"] or rec["grid_candidates_evaluated"]!=prior["grid_candidates_evaluated"]:
                raise ValueError(
                    "The layer grids differ. Use identical search ranges "
                    "for a fair comparison.")
            if rec["validation_images"]!=prior["validation_images"] or rec["validation_errors"]!=prior["validation_errors"]:
                raise ValueError("Validation counts differ between layers")
            verify_aligned(baseline,data,name,a.logit_atol)
        else:
            baseline=data
        candidates.append(rec)
    selected=max(candidates,
                 key=lambda x:(x["selected_config"]["auroc"],
                               x["selected_config"]["auprc"]))
    report={
        "protocol":("post_test_exploratory_layer_screen"
                    if a.post_test_exploratory
                    else "validation_only_prospective_layer_screen"),
        "selection_rule":"Max VAL error-AUROC then AUPRC using identical full-topic grid for every layer",
        "manifest_sha256":manifest_sha,
        "checkpoint_sha256":selected["checkpoint_sha256"],
        "candidates":candidates,
        "selected_layer":selected["layer"],
        "selected_arrays_root":selected["arrays_root"],
        "selected_grid_json":selected["grid_json"],
        "selected_configuration":selected["selected_config"],
        "caution":(
            "TEST previously examined: this selection is exploratory and "
            "must not replace the original frozen-test estimate as an "
            "independent confirmation." if a.post_test_exploratory else
            "Freeze this validation choice before opening TEST; do not "
            "retune the layer after test inspection."
        )
    }
    output=Path(a.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Selection protocol: {report['protocol']}")
    for r in candidates:
        w=r["selected_config"]
        print(f"  {r['name']} {r['layer']}: VAL AUROC {w['auroc']:.6f}, "
              f"AUPRC {w['auprc']:.6f}; K={w['K']} M={w['M']} J={w['J']}")
    print("Best layer:",report["selected_layer"])
    print("Arrays root:",report["selected_arrays_root"])
    print("Grid JSON:",report["selected_grid_json"])
    print("Note:",report["caution"])
    print("Saved:",output)


if __name__=="__main__":
    main()
