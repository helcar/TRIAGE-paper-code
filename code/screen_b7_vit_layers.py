"""Fixed-configuration validation-only ViT layer screening.

Each candidate uses the same K/M/J/tau/weights, FIT/CAL/VAL partition,
bounded feature reservoir, and seed. TEST is never accessed.
"""
import argparse
import csv
import json
from pathlib import Path
from dataclasses import asdict
from .triage import Config
from .validate_triage import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-prefix",required=True,
                   help="D:/TRIAGE/vit_ffpp_finetuned_layer (suffix: 2,5,7,9,11)_arrays")
    p.add_argument("--layers",nargs="+",type=int,default=[2,5,7,9,11])
    p.add_argument("--output",required=True)
    p.add_argument("--output-csv",required=True)
    p.add_argument("--max-local-features",type=int,default=50000)
    p.add_argument("--transform-batch",type=int,default=32)
    a=p.parse_args()
    cfg=Config(prototypes=128,topics_positive=16,topics_negative=16,
               references_real=2,references_fake=2,tau=1.,
               lambda_support=1.,lambda_oppose=1.,lambda_magnitude=.1,
               seed=42,max_local_features=a.max_local_features)
    records=[]
    for layer in a.layers:
        root=Path(a.root_prefix+str(layer)+"_arrays")
        for split in ("fit","cal","val"):
            if not (root/split/"_complete.json").is_file():
                p.error(f"Incomplete layer {layer} {split}: {root}")
        result=run(root/"fit",root/"cal",root/"val",cfg,a.transform_batch)
        record={"layer":layer,"n":result["validation_images"],
                "errors":result["validation_errors"],
                "triage_auroc":result["TRIAGE"]["AUROC"],
                "triage_auprc":result["TRIAGE"]["AUPRC"],
                "confidence_auroc":result["Confidence"]["AUROC"],
                "confidence_auprc":result["Confidence"]["AUPRC"]}
        records.append(record)
        print(json.dumps(record),flush=True)
    if len({(r["n"],r["errors"]) for r in records})!=1:
        raise RuntimeError("Detector outcomes differ across layers; inspect array provenance")
    best=max(records,key=lambda r:(r["triage_auroc"],r["triage_auprc"]))
    report={"protocol":"VAL-only fixed-configuration layer screening; TEST unopened",
            "config":asdict(cfg),"layers":records,"selected_layer":best["layer"],
            "selection_rule":"Highest TRIAGE validation AUROC, then AUPRC",
            "selected":best}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    dest=Path(a.output_csv);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)
    print("SELECTED",json.dumps(best),flush=True)


if __name__=="__main__":
    main()
