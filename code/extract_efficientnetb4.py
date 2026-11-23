"""Stream EfficientNet-B4 frozen detector outputs for TRIAGE.

Reuses the SAME manifest, pre-cropped RGB inputs and [-1,1] normalization
as Xception. Produces <out>/<split>/{logits,labels,features,attributions}.npy
compatible with code.run_experiment.read, with a provenance-bound marker.

The TEST split is intentionally opt-in. Fit/cal/val extraction and model
selection should finish before test scores are consulted.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .extract_ffpp import ManifestFrames, load_manifest
from .triage import DetectorAttributions
from .efficientnetb4_adapter import (
    build_deepfakebench_efficientnetb4, resolve_layer)


ARRAYS=("logits","labels","features","attributions")


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(8*1024*1024),b""):
            digest.update(part)
    return digest.hexdigest()


def extract_one(rows,split,out,root,extractor,batch_size,image_size,
                provenance,overwrite=False):
    out=Path(out)
    meta={**provenance,"split":split,
          "samples":len(rows),"num_real":sum(r["label"]=="0" for r in rows),
          "num_fake":sum(r["label"]=="1" for r in rows)}
    flag=out/"_complete.json"
    if flag.exists() and not overwrite:
        old=json.loads(flag.read_text(encoding="utf-8"))
        for key in meta:
            if old.get(key)!=meta[key]:
                raise RuntimeError(
                    f"Existing {out} has a different {key}. "
                    "Do not mix detectors/manifests/layers. "
                    "Use a separate output folder.")
        if all((out/(n+".npy")).is_file() for n in ARRAYS):
            print(f"SKIP already complete {split}, n={len(rows)}",flush=True)
            return
        raise RuntimeError(f"Broken extraction at {out}")
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"{out} contains a partial extraction. Rerun with "
                "--overwrite to start this split from scratch.")
        shutil.rmtree(out)
    if not rows:
        raise ValueError(f"No manifest frames for {split}")
    out.mkdir(parents=True,exist_ok=True)
    loader=DataLoader(ManifestFrames(rows,root,image_size),
                      batch_size=batch_size,shuffle=False,num_workers=0)
    buffers={}
    offset=0
    for images,labels in loader:
        logits,features,attributions=extractor.extract(images)
        values={
            "logits":np.asarray(logits,dtype=np.float32),
            "features":np.asarray(features,dtype=np.float32),
            "attributions":np.asarray(attributions,dtype=np.float32),
            "labels":labels.numpy().astype(np.int64),
        }
        if not buffers:
            for k,v in values.items():
                buffers[k]=np.lib.format.open_memmap(
                    out/(k+".npy"),mode="w+",dtype=v.dtype,
                    shape=(len(rows),*v.shape[1:]))
            meta["features_shape"]=list(buffers["features"].shape)
            meta["attributions_shape"]=list(buffers["attributions"].shape)
        size=len(labels)
        for k,v in values.items():
            buffers[k][offset:offset+size]=v
        offset+=size
        if offset%200<size:
            print(f"{split}: {offset}/{len(rows)}",flush=True)
    if offset!=len(rows):
        raise RuntimeError(f"Extracted {offset} of {len(rows)} {split} frames")
    for v in buffers.values():
        v.flush()
    del buffers
    flag.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"complete":str(out),"samples":offset,
                      "shape":meta["features_shape"]}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--image-root",default=".")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--bench-dir",required=True)
    p.add_argument("--checkpoint",required=True,
                   help="Deepfake-trained B4 weights, NOT ImageNet pretrained")
    p.add_argument("--layer",default="backbone.efficientnet._blocks.21")
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--image-size",type=int,default=256)
    p.add_argument("--splits",choices=("fit","cal","val","test"),nargs="+",
                   default=("fit","cal","val"))
    p.add_argument("--authorize-test",action="store_true",
                   help="Explicit opt-in to reading TEST only after validation selection")
    p.add_argument("--overwrite",action="store_true")
    a=p.parse_args()
    if a.batch_size<1 or a.image_size<1:
        p.error("Batch size and image size must be positive")
    if len(set(a.splits))!=len(a.splits):
        p.error("Duplicate split")
    if "test" in a.splits and not a.authorize_test:
        p.error("TEST is locked. Complete validation-only selection, "
                "then add --authorize-test explicitly.")
    target=Path(a.output_dir).expanduser().resolve()
    if "efficientnet" not in str(target).lower():
        p.error("Use an EfficientNet-named output folder so these arrays "
                "cannot be confused with Xception.")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    manifest=Path(a.manifest).expanduser().resolve()
    weights=Path(a.checkpoint).expanduser().resolve()
    rows=load_manifest(manifest)
    model,loadinfo=build_deepfakebench_efficientnetb4(
        a.bench_dir,weights,a.device)
    prov={
        "detector":"DeepfakeBench EfficientNetB4",
        "checkpoint_sha256":sha256_file(weights),
        "manifest_sha256":sha256_file(manifest),
        "layer":a.layer,
        "image_size":a.image_size,
        "normalization":"(RGB/255-0.5)/0.5",
        "loaded_tensors":loadinfo["loaded_tensors"],
    }
    print("Frozen backbone, metadata:",json.dumps(prov),flush=True)
    extractor=DetectorAttributions(
        model,resolve_layer(model,a.layer),device=a.device)
    try:
        for split in a.splits:
            selected=[r for r in rows if r["split"]==split]
            extract_one(selected,split,target/split,a.image_root,
                        extractor,a.batch_size,a.image_size,
                        prov,a.overwrite)
    finally:
        extractor.close()


if __name__=="__main__":
    main()
