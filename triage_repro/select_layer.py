"""Select one detector feature layer exclusively from validation results.

Each candidate must have been tuned using the same search grid and evaluated
on aligned frames with an identical frozen detector checkpoint. No TEST read.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def choose(candidates, output, tolerance=0.005):
    if len(candidates) < 2:
        raise ValueError("Layer screen requires at least two candidates")
    entries = []
    reference = None
    ref_logits = None
    for layer, root, grid_file in candidates:
        root = Path(root)
        report = json.loads(Path(grid_file).read_text(encoding="utf-8"))
        if report["best"]["full_topic"]["kind"] != "full_topic":
            raise ValueError("Expected validation-selected full-topic settings")
        provenance = None
        for part in ("fit", "cal", "val"):
            marker = root / part / "_complete.json"
            if not marker.is_file():
                raise FileNotFoundError(marker)
            meta = json.loads(marker.read_text(encoding="utf-8"))
            if meta.get("layer") != layer:
                raise ValueError(f"{root}: layer provenance mismatch")
            actual = (meta.get("checkpoint_sha256"), meta.get("manifest_sha256"),
                      meta.get("image_size"), meta.get("normalization"))
            if not actual[0]:
                raise ValueError("Missing checkpoint SHA")
            if provenance is None:
                provenance = actual
            if actual != provenance:
                raise ValueError(f"{root}: inconsistent detector across splits")
        if reference is None:
            reference = (provenance, report["search"])
        elif (provenance, report["search"]) != reference:
            raise ValueError("Layer candidates use different detector, manifest, or search grid")
        val_logits = np.load(root / "val" / "logits.npy", mmap_mode="r")
        val_labels = np.load(root / "val" / "labels.npy", mmap_mode="r")
        if ref_logits is None:
            ref_logits = (np.array(val_logits), np.array(val_labels))
        elif (not np.array_equal(val_labels, ref_logits[1]) or
              not np.allclose(val_logits, ref_logits[0], atol=tolerance, rtol=1e-5)):
            raise ValueError(f"{layer}: detector validation logits or labels differ")
        best = report["best"]["full_topic"]
        entries.append({
            "layer": layer, "arrays_root": str(root),
            "grid_json": str(grid_file),
            "auroc": float(best["auroc"]), "auprc": float(best["auprc"]),
        })
    selected = max(entries, key=lambda x: (x["auroc"], x["auprc"]))
    result = {
        "selected_layer": selected["layer"],
        "selected_arrays_root": selected["arrays_root"],
        "selected_grid_json": selected["grid_json"],
        "selection_rule": "Highest validation error-ranking AUROC; AUPRC tie-breaker",
        "test_accessed": False,
        "candidates": entries,
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite layer selection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", nargs=3, action="append", required=True,
                   metavar=("LAYER", "ARRAYS_DIR", "GRID_JSON"))
    p.add_argument("--output", required=True)
    p.add_argument("--logit-atol", type=float, default=0.005)
    args = p.parse_args()
    print(json.dumps(choose(args.candidate, args.output, args.logit_atol), indent=2))


if __name__ == "__main__":
    main()
