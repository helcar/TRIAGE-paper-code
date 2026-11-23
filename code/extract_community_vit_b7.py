"""Extract frozen CommunityForensics ViT arrays for TRIAGE B7.

Strictly loads the exact detector checkpoint and uses original HF preprocessing.
FIT/CAL/VAL first; TEST requires --authorize-test. No label-dependent fitting.
Output matches code.run_experiment.read() directory layout.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .triage import DetectorAttributions
from .extract_ffpp import load_manifest
from .preflight_community_vit_b7 import MODEL_ID, BinaryAdapter

FIELDS = ("logits", "labels", "features", "attributions")


def load_detector(revision, device):
    from transformers import AutoConfig, ViTForImageClassification, ViTImageProcessor
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    kw = {"trust_remote_code": False}
    if revision:
        kw["revision"] = revision
    config = AutoConfig.from_pretrained(MODEL_ID, **kw)
    if not (config.model_type == "vit" and config.num_labels == 1
            and config.num_attention_heads == 6
            and config.intermediate_size == 1536):
        raise RuntimeError("Incorrect/legacy checkpoint config; do not run B7")
    processor = ViTImageProcessor.from_pretrained(MODEL_ID, **kw)
    weights = hf_hub_download(MODEL_ID, "model.safetensors",
                              revision=revision)
    model = ViTForImageClassification(config)
    from safetensors.torch import load_file
    state = load_file(weights, device="cpu")
    # Avoid relying on lenient from_pretrained() behavior.
    # The repository stores pre-Transformers-5 ViT encoder names.
    # Map each parameter exactly; retain strict=True to detect incomplete weights.
    import re
    migrated = {}
    aliases = {
        "attention.attention.query": "attention.q_proj",
        "attention.attention.key": "attention.k_proj",
        "attention.attention.value": "attention.v_proj",
        "attention.output.dense": "attention.o_proj",
        "intermediate.dense": "mlp.fc1",
        "output.dense": "mlp.fc2",
    }
    for key, value in state.items():
        hit = re.fullmatch(r"vit\.encoder\.layer\.(\d+)\.(.+)", key)
        if hit:
            suffix = hit.group(2)
            for original, updated in aliases.items():
                if suffix.startswith(original + "."):
                    suffix = updated + suffix[len(original):]
                    break
            key = f"vit.layers.{hit.group(1)}.{suffix}"
        if key in migrated:
            raise RuntimeError(f"Duplicate migrated checkpoint key: {key}")
        migrated[key] = value
    model.load_state_dict(migrated, strict=True)
    print("Strictly loaded corrected CommunityForensics ViT weights.", flush=True)
    detector = BinaryAdapter(model).to(device).eval()
    for par in detector.parameters():
        par.requires_grad_(False)

    return detector, processor, weights


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--training-manifest", default=None,
                   help="Original FF++ detector-training manifest when extracting other datasets")
    p.add_argument("--finetuned-checkpoint", default=None,
                   help="Local FF++ ViT training checkpoint; never load for the old baseline output directory")
    p.add_argument("--layer", default="classifier.vit.layers.7")
    p.add_argument("--device", default="cuda")
    p.add_argument("--splits", nargs="+", choices=("fit", "cal", "val", "test"),
                   default=("fit", "cal", "val"))
    p.add_argument("--authorize-test", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if "test" in a.splits and not a.authorize_test:
        p.error("TEST is locked until validation-only configuration selection.")
    if len(a.splits) != len(set(a.splits)):
        p.error("Duplicate split requested")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    if "vit" not in a.output_dir.lower():
        p.error("Output directory name must contain 'vit' to avoid CNN array mixing")

    manifest = Path(a.manifest).resolve()
    rows = load_manifest(manifest)
    detector, processor, checkpoint = load_detector(a.revision, a.device)
    if a.finetuned_checkpoint:
        local = Path(a.finetuned_checkpoint).resolve()
        if not local.is_file():
            p.error(f"Missing fine-tuned checkpoint: {local}")
        state = torch.load(local, map_location="cpu", weights_only=False)
        if state.get("base_checkpoint_sha256") != sha256(checkpoint):
            raise RuntimeError("Base-checkpoint provenance mismatch")
        training_manifest = Path(a.training_manifest or a.manifest).resolve()
        if state.get("manifest_sha256") != sha256(training_manifest):
            raise RuntimeError("Training manifest provenance mismatch; supply --training-manifest for cross-dataset extraction")
        detector.classifier.load_state_dict(state["state_dict"], strict=True)
        detector.eval()
        checkpoint = local
        print(f"Strictly loaded local FF++ ViT checkpoint (epoch {state['epoch']}).", flush=True)
    modules = dict(detector.named_modules())
    if a.layer not in modules:
        raise ValueError(f"Unknown transformer layer: {a.layer}")
    provenance = {
        "detector": MODEL_ID + (" (FF++-fine-tuned)" if a.finetuned_checkpoint else ""),
        "checkpoint_sha256": sha256(checkpoint),
        "finetuned_checkpoint": bool(a.finetuned_checkpoint),
        "manifest_sha256": sha256(manifest),
        "layer": a.layer,
        "revision": a.revision,
        "preprocessing": processor.to_dict(),
        "excluded_cls_token": True,
    }
    extractor = DetectorAttributions(detector, modules[a.layer],
                                     device=a.device, exclude_cls_token=True)
    try:
        for split in a.splits:
            selected = [row for row in rows if row["split"] == split]
            if not selected:
                raise ValueError(f"No manifest rows for {split}")
            target = Path(a.output_dir) / split
            meta = dict(provenance, split=split, samples=len(selected))
            marker = target / "_complete.json"
            if marker.is_file() and not a.overwrite:
                old = json.loads(marker.read_text(encoding="utf-8"))
                # Normalize tuples / nested processor metadata exactly as
                # json.dump did when writing the original marker.
                comparable = json.loads(json.dumps(meta))
                mismatched = [key for key, val in comparable.items()
                              if old.get(key) != val]
                missing_files = [key for key in FIELDS
                                 if not (target / (key + ".npy")).is_file()]
                if not mismatched and not missing_files:
                    # Verify on-disk shapes before considering the split complete.
                    wrong_shapes = []
                    for key in FIELDS:
                        arr = np.load(target / (key + ".npy"), mmap_mode="r")
                        if len(arr) != len(selected):
                            wrong_shapes.append(
                                f"{key}: {len(arr)} != {len(selected)}")
                        del arr
                    if not wrong_shapes:
                        print(f"SKIP complete {split}: {len(selected)}", flush=True)
                        continue
                    raise RuntimeError(
                        f"Existing {target} has invalid array shapes: {wrong_shapes}")
                raise RuntimeError(
                    f"Existing {target} cannot be reused: "
                    f"mismatched metadata keys={mismatched}, "
                    f"missing arrays={missing_files}. "
                    "Do not use --overwrite until the differences are reviewed.")
            if target.exists() and any(target.iterdir()):
                if not a.overwrite:
                    raise RuntimeError(f"Partial {target}; use --overwrite if intentional")
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            buffers = {}
            try:
                for index, row in enumerate(selected):
                    filename = Path(row["path"])
                    if not filename.is_absolute():
                        filename = Path(a.image_root) / filename
                    with Image.open(filename) as im:
                        pixels = processor(images=im.convert("RGB"),
                                           return_tensors="pt")["pixel_values"]
                    logit, features, attribution = extractor.extract(pixels)
                    values = {
                        "logits": np.asarray(logit, dtype=np.float32),
                        "labels": np.asarray([int(row["label"])], dtype=np.int64),
                        "features": np.asarray(features, dtype=np.float32),
                        "attributions": np.asarray(attribution, dtype=np.float32),
                    }
                    if not all(np.isfinite(arr).all() for arr in values.values()):
                        raise RuntimeError(f"Nonfinite array for {filename}")
                    if not buffers:
                        for key, arr in values.items():
                            buffers[key] = np.lib.format.open_memmap(
                                target / (key + ".npy"), mode="w+", dtype=arr.dtype,
                                shape=(len(selected), *arr.shape[1:]))
                        meta["feature_shape"] = list(buffers["features"].shape)
                    for key, value in values.items():
                        buffers[key][index] = value[0]
                    if (index + 1) % 200 == 0:
                        print(f"{split}: {index+1}/{len(selected)}", flush=True)
                for buffer in buffers.values():
                    buffer.flush()
                # Retain the dictionary until the finally block finishes;
                # deleting it here caused UnboundLocalError after completion.
                marker.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                print(f"COMPLETE {split}: {len(selected)}; {meta['feature_shape']}", flush=True)
            finally:
                if buffers:
                    for buffer in buffers.values():
                        try:
                            buffer.flush()
                        except Exception:
                            pass
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
