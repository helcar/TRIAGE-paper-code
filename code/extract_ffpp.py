"""Extract TRIAGE arrays from an approved, pre-cropped FaceForensics++ frame manifest.

Input CSV columns: path,label,split,group_id
 - path: absolute path or relative to --image-root
 - label: 0 real, 1 fake
 - split: fit/cal/val/test, assigned BEFORE frame extraction
 - group_id: source-video / identity group; never shared across splits
Images MUST already be face-cropped consistently with DeepfakeBench's
training preprocessing. This script does not perform face detection.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .triage import DetectorAttributions
from .xception_adapter import build_deepfakebench_xception, resolve_layer


class ManifestFrames(Dataset):
    def __init__(self, rows, root, size):
        self.rows, self.root, self.size = rows, Path(root), size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as img:
            img = img.convert("RGB").resize((self.size, self.size),
                                              Image.Resampling.BILINEAR)
            array = np.asarray(img, dtype=np.float32) / 255.
        # DeepfakeBench commonly normalizes Xception inputs to [-1, 1].
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        tensor = (tensor - 0.5) / 0.5
        return tensor, int(row["label"])


def load_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"path", "label", "split", "group_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest requires columns {sorted(required)}")
    groups = {}
    seen = set()
    for row in rows:
        split = row["split"]
        if split not in {"fit", "cal", "val", "test"}:
            raise ValueError(f"Invalid split {split}")
        if row["label"] not in {"0", "1"}:
            raise ValueError("Labels must be 0 or 1")
        if not row["group_id"]:
            raise ValueError("group_id is required")
        previous = groups.setdefault(row["group_id"], split)
        if previous != split:
            raise ValueError(f"Leakage: group {row['group_id']} occurs in {previous} and {split}")
        pair = (split, row["path"])
        if pair in seen:
            raise ValueError(f"Repeated frame: {pair}")
        seen.add(pair)
    return rows


def extract(rows, root, out_dir, model, layer, device, batch_size, size):
    extractor = DetectorAttributions(model, layer, device=device)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for split in ("fit", "cal", "val", "test"):
            selected = [r for r in rows if r["split"] == split]
            if not selected:
                raise ValueError(f"No rows for split {split}")
            dataset = ManifestFrames(selected, root, size)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0)
            all_logits, all_features, all_attr, all_y = [], [], [], []
            for batch, labels in loader:
                logits, features, attributions = extractor.extract(batch)
                all_logits.append(logits.astype(np.float32))
                all_features.append(features.astype(np.float32))
                all_attr.append(attributions.astype(np.float32))
                all_y.append(labels.numpy().astype(np.int64))
            output = out_dir / f"{split}.npz"
            np.savez_compressed(output,
                logits=np.concatenate(all_logits),
                labels=np.concatenate(all_y),
                features=np.concatenate(all_features),
                attributions=np.concatenate(all_attr))
            print(json.dumps({"split":split,"images":len(selected),
                              "output":str(output),"labels":np.bincount(
                                  np.concatenate(all_y),minlength=2).tolist()}))
    finally:
        extractor.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--output-dir", default="data/npz")
    p.add_argument("--bench-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--layer", default="backbone.block12")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    rows = load_manifest(args.manifest)
    model, info = build_deepfakebench_xception(
        args.bench_dir, args.checkpoint, args.device)
    print("Checkpoint:", json.dumps(info))
    extract(rows, args.image_root, Path(args.output_dir), model,
            resolve_layer(model, args.layer), args.device,
            args.batch_size, args.image_size)


if __name__ == "__main__":
    main()
