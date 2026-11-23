"""Stream Xception outputs into on-disk .npy arrays; resume completed splits.

Use a manifest from prepare_ffpp_c23_manifest.py and checkpoint-matched
aligned face crops. Writes each split to OUTPUT_DIR/<split>/{logits,labels,
features,attributions}.npy. Arrays can be read with mmap_mode='r'.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .extract_ffpp import ManifestFrames, load_manifest
from .triage import DetectorAttributions
from .xception_adapter import build_deepfakebench_xception, resolve_layer

ARRAY_NAMES=("logits","labels","features","attributions")


def extract_split(rows, root, out, extractor, batch_size, size, overwrite=False):
    out=Path(out)
    flag=out / "_complete.json"
    if flag.exists() and not overwrite:
        meta=json.loads(flag.read_text())
        if meta.get("images")==len(rows) and all((out/f"{name}.npy").is_file() for name in ARRAY_NAMES):
            print(json.dumps({"skipped_completed":str(out),"images":len(rows)}))
            return
        raise RuntimeError(f"Existing output does not match manifest: {out}. Delete it before retry")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise RuntimeError(f"Incomplete data in {out}; remove directory or use --overwrite")
    if len(rows)==0: raise ValueError(f"Empty split {out.name}")
    out.mkdir(parents=True,exist_ok=True)
    dataset=ManifestFrames(rows,root,size)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    buffers={}
    offset=0
    try:
        for images,labels in loader:
            logits,feat,attr=extractor.extract(images)
            batch={"logits":np.asarray(logits,dtype=np.float32),
                   "features":np.asarray(feat,dtype=np.float32),
                   "attributions":np.asarray(attr,dtype=np.float32),
                   "labels":labels.numpy().astype(np.int64)}
            if not buffers:
                for name,array in batch.items():
                    buffers[name]=np.lib.format.open_memmap(
                        out/f"{name}.npy",mode="w+",dtype=array.dtype,
                        shape=(len(rows),*array.shape[1:]))
            end=offset+len(labels)
            for name,array in batch.items(): buffers[name][offset:end]=array
            offset=end
            if offset % 1000 < len(labels):
                print(json.dumps({"split":out.name,"processed":offset,"total":len(rows)}),flush=True)
        if offset!=len(rows): raise RuntimeError(f"Processed {offset} of {len(rows)}")
        for arr in buffers.values(): arr.flush()
        del buffers
        flag.write_text(json.dumps({"images":len(rows),"labels":{
            "real":sum(x["label"]=="0" for x in rows),
            "fake":sum(x["label"]=="1" for x in rows)}}))
        print(json.dumps({"complete":str(out),"images":offset}),flush=True)
    except Exception:
        # No completion marker means this split must be rerun.
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--image-root",default=".")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--bench-dir",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--layer",default="backbone.block12")
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--image-size",type=int,default=256)
    p.add_argument("--device",default="cuda")
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--splits",nargs="+",choices=("fit","cal","val","test"),default=("fit","cal","val","test"),help="Choose splits; omit test during exploratory development")
    args=p.parse_args()
    if args.batch_size<1: p.error("batch-size must be positive")
    rows=load_manifest(args.manifest)
    output=Path(args.output_dir)
    model, info=build_deepfakebench_xception(args.bench_dir,args.checkpoint,args.device)
    print("Checkpoint:",json.dumps(info))
    extractor=DetectorAttributions(model,resolve_layer(model,args.layer),device=args.device)
    try:
        for split in args.splits:
            selected=[r for r in rows if r["split"]==split]
            extract_split(selected,args.image_root,output/split,extractor,
                          args.batch_size,args.image_size,args.overwrite)
    finally:
        extractor.close()


if __name__=="__main__":
    main()
