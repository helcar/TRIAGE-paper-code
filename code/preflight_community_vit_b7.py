"""Validation-only B7 preflight: CommunityForensics ViT-Small (one-logit fake score).

Distinct architecture/provenance from the previous two-logit ViT candidate.
Require corrected checkpoint config (2026 fix) and exact weights.
Uses published Hugging Face image processor, never Xception normalization.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from .triage import DetectorAttributions


MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"


class BinaryAdapter(torch.nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, images):
        logits = self.classifier(pixel_values=images).logits
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise ValueError(f"Expected a single fake-class logit: {logits.shape}")
        return logits[:, 0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--revision", default=None,
                   help="Optional fixed Hugging Face model commit SHA")
    p.add_argument("--layer", default="classifier.vit.layers.7")
    p.add_argument("--max-per-class", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="results/b7_community_vit_preflight.json")
    a = p.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA unavailable; use --device cpu")
    if a.max_per_class < 2:
        p.error("--max-per-class must be at least 2")

    from transformers import AutoConfig, ViTForImageClassification, ViTImageProcessor
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    kw = {"trust_remote_code": False}
    if a.revision:
        kw["revision"] = a.revision
    config = AutoConfig.from_pretrained(MODEL_ID, **kw)
    if not (config.model_type == "vit" and config.num_labels == 1
            and config.num_attention_heads == 6
            and config.intermediate_size == 1536):
        raise RuntimeError("Incorrect/legacy checkpoint config; do not run B7")
    processor = ViTImageProcessor.from_pretrained(MODEL_ID, **kw)
    weights = hf_hub_download(MODEL_ID, "model.safetensors",
                              revision=a.revision)
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
    detector = BinaryAdapter(model).to(a.device).eval()
    for par in detector.parameters():
        par.requires_grad_(False)

    layers = dict(detector.named_modules())
    if a.layer not in layers:
        raise ValueError(f"Unknown {a.layer}; example modules: "
                         f"{[k for k in layers if 'vit.layers' in k][-10:]}")

    with open(a.manifest, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    picked = []
    for cls in (0, 1):
        subset = [r for r in rows if int(r["label"]) == cls]
        if len(subset) < a.max_per_class:
            p.error(f"Only {len(subset)} validation images for class {cls}")
        picked += subset[:a.max_per_class]
    extractor = DetectorAttributions(detector, layers[a.layer],
                                     device=a.device, exclude_cls_token=True)
    labels, logits, max_attr = [], [], []
    feature_shape = None
    try:
        for row in picked:
            filename = Path(row["path"])
            if not filename.is_absolute():
                filename = Path(a.image_root) / filename
            with Image.open(filename) as im:
                pixels = processor(images=im.convert("RGB"),
                                   return_tensors="pt")["pixel_values"]
            z, features, signed = extractor.extract(pixels)
            if not np.isfinite(z).all() or not np.isfinite(features).all() or not np.isfinite(signed).all():
                raise RuntimeError(f"Nonfinite features/attributions: {filename}")
            feature_shape = list(features.shape)
            labels.append(int(row["label"]))
            logits.append(float(z[0]))
            max_attr.append(float(np.abs(signed).max()))
    finally:
        extractor.close()
    result = {
        "model": MODEL_ID,
        "revision": a.revision,
        "checkpoint_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        "layer": a.layer,
        "samples": len(labels),
        "feature_shape_without_cls": feature_shape,
        "validation_detector_auroc": float(roc_auc_score(labels, logits)),
        "predicted_fake": int(np.count_nonzero(np.asarray(logits) >= 0)),
        "attribution_median_maxabs": float(np.median(max_attr)),
        "attribution_maxabs": float(np.max(max_attr)),
        "nonzero_attribution_fraction": float(np.mean(np.asarray(max_attr) > 1e-10)),
        "note": "Validation-only transfer check; AI-generated-image detector, not necessarily facial deepfake-trained."
    }
    target = Path(a.output)
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
