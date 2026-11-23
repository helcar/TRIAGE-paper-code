"""Benchmark frozen Xception forward and gradient-times-activation extraction.

Uses exactly the same checkpoint adapter and image normalization as the
experiment extractor. Times include detector computation plus tensor output
transfer to host for gradient extraction, but exclude image decoding.
No changes to detector weights. Synchronizes CUDA before and after timing.
"""
import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from .extract_ffpp import ManifestFrames
from .triage import DetectorAttributions
from .xception_adapter import build_deepfakebench_xception, resolve_layer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--image-root", default=".",
                   help="Base folder for relative manifest image paths")
    p.add_argument("--bench-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layer", default="backbone.block8")
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    a=p.parse_args()
    if a.batch_size < 1 or a.repeats < 1 or a.warmup < 0:
        p.error("Invalid timing repetitions or batch size")
    with open(a.manifest, newline="", encoding="utf-8-sig") as f:
        manifest=[r for r in csv.DictReader(f) if r["split"]=="test"]
    if len(manifest)<a.batch_size:
        p.error("Not enough test images")
    # Preload a fixed small batch so disk I/O does not enter GPU timings.
    dataset=ManifestFrames(manifest[:a.batch_size], a.image_root, a.image_size)
    batch=torch.stack([dataset[i][0] for i in range(a.batch_size)])
    model, info=build_deepfakebench_xception(a.bench_dir,a.checkpoint,a.device)
    x=batch.to(a.device)
    extractor=None
    def sync():
        if x.device.type=="cuda":
            torch.cuda.synchronize()
    def forward():
        with torch.no_grad():
            out=model(x)
            if torch.is_tensor(out):
                _=out.detach()
    def attribution():
        extractor.extract(x)
    def measure(fn):
        values=[]
        for _ in range(a.warmup):
            fn()
        sync()
        for _ in range(a.repeats):
            sync()
            start=time.perf_counter()
            fn()
            sync()
            values.append((time.perf_counter()-start)*1000/a.batch_size)
        return {"mean_ms_per_image":float(statistics.mean(values)),
                "median_ms_per_image":float(statistics.median(values)),
                "min_ms_per_image":float(min(values)),
                "max_ms_per_image":float(max(values))}
    # Install the gradient hook only after the no-grad forward benchmark.
    detector_forward=measure(forward)
    extractor=DetectorAttributions(model,resolve_layer(model,a.layer),a.device)
    try:
        grad_act=measure(attribution)
    finally:
        extractor.close()
    perf={"detector_forward":detector_forward,
          "forward_plus_grad_act":grad_act}
    result={"checkpoint":a.checkpoint,"loaded_tensors":info["loaded_tensors"],
            "layer":a.layer,"device":a.device,"gpu":
            torch.cuda.get_device_name(0) if x.device.type=="cuda" else None,
            "python":platform.python_version(),"torch":torch.__version__,
            "batch_size":a.batch_size,"image_size":a.image_size,
            "warmup":a.warmup,"repeats":a.repeats,"timings":perf,
            "benchmark_scope":"preloaded image tensor; excludes face detection and image decoding",
            "note":"forward_plus_grad_act includes CPU transfer of logits/features/attributions; "
                   "detector_forward excludes CPU output transfer"}
    result["approx_grad_act_extra_ms_per_image"]=(
        perf["forward_plus_grad_act"]["mean_ms_per_image"] -
        perf["detector_forward"]["mean_ms_per_image"])
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,default=str))
    print(f"Saved {dest}")


if __name__=="__main__":
    main()
