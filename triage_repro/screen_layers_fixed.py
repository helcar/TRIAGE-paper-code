"""Final-paper fixed-configuration VAL layer screen for Xception and ViT.

Xception and ViT use one common TRIAGE setting at each candidate layer,
choose highest VAL AUROC (AUPRC tie-breaker), then tune only that layer.
EfficientNet-B4 uses the full-grid layer screen in select_layer.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from code.triage import Config
from code.validate_triage import run as validate_fixed


def screen(candidates, output, batch=32, atol=0.005):
    if not candidates:
        raise ValueError("No layers to screen")
    config = Config(
        prototypes=128, topics_positive=16, topics_negative=16,
        references_real=2, references_fake=2, tau=1.0,
        lambda_support=1.0, lambda_oppose=1.0, lambda_magnitude=0.1,
        max_local_features=50000, seed=42,
    )
    records = []
    reference = None
    reference_logits = None
    for layer, root in candidates:
        root = Path(root)
        provenance = None
        for split in ("fit", "cal", "val"):
            marker = root / split / "_complete.json"
            if not marker.exists():
                raise FileNotFoundError(marker)
            meta = json.loads(marker.read_text(encoding="utf-8"))
            if meta.get("layer") != layer:
                raise ValueError(f"{root}: wrong layer in {split} metadata")
            current = (meta.get("checkpoint_sha256"),
                       meta.get("manifest_sha256"),
                       meta.get("image_size"), meta.get("normalization"))
            if not current[0]:
                raise ValueError("Missing detector checkpoint hash")
            if provenance is None:
                provenance = current
            if current != provenance:
                raise ValueError(f"Detector changed across {root} splits")
        if reference is None:
            reference = provenance
        elif provenance != reference:
            raise ValueError(f"Detector or frame manifest changed between candidate layers: {layer}")
        logits = np.load(root / "val" / "logits.npy", mmap_mode="r")
        labels = np.load(root / "val" / "labels.npy", mmap_mode="r")
        if reference_logits is None:
            reference_logits = (np.asarray(logits).copy(), np.asarray(labels).copy())
        elif (not np.array_equal(labels, reference_logits[1])
              or not np.allclose(logits, reference_logits[0], atol=atol, rtol=1e-5)):
            raise ValueError(f"Validation logits/labels differ for {layer}")
        result = validate_fixed(root / "fit", root / "cal", root / "val",
                                config, transform_batch=batch)
        records.append({
            "layer": layer,
            "arrays_root": str(root),
            "validation_auroc": result["TRIAGE"]["AUROC"],
            "validation_auprc": result["TRIAGE"]["AUPRC"],
            "validation_images": result["validation_images"],
            "validation_errors": result["validation_errors"],
        })
    best = max(records, key=lambda record: (
        record["validation_auroc"], record["validation_auprc"]))
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "selected_layer": best["layer"],
        "selected_arrays_root": best["arrays_root"],
        "selected_grid_json": str(Path(best["arrays_root"]) / "validation_grid.json"),
        "selection_rule": "Fixed common (K=128,M=16,J=2,tau=1,weights=1/1/0.1) VAL AUROC; AUPRC tie-breaker",
        "fixed_configuration": {
            "K": 128, "M": 16, "J": 2, "tau": 1.0,
            "lambda_support": 1.0, "lambda_oppose": 1.0,
            "lambda_magnitude": 0.1, "max_local_features": 50000,
            "seed": 42, "transform_batch": batch,
        },
        "test_accessed": False,
        "candidates": records,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", action="append", nargs=2, required=True,
                   metavar=("LAYER", "ARRAYS_ROOT"))
    p.add_argument("--output", required=True)
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    print(json.dumps(screen(a.candidate, a.output, a.batch), indent=2))


if __name__ == "__main__":
    main()
