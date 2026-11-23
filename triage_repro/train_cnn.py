"""Fine-tune DeepfakeBench Xception/EfficientNet-B4 from ImageNet weights.

This is a NEW prospective experiment, not reconstruction of historical CNN
training. Never initialize from an FF++-trained DeepfakeBench detector file.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .manifest import audit, load_manifest, sha256


class FaceFrames(Dataset):
    def __init__(self, rows, image_root, size, mean, std, flip_probability=0.0):
        self.rows = rows
        self.image_root = Path(image_root)
        self.size = int(size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.flip_probability = float(flip_probability)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.image_root / path
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if image.size != (self.size, self.size):
                raise ValueError(f"Expected aligned {self.size}x{self.size} crop: {path} is {image.size}")
            if self.flip_probability and random.random() < self.flip_probability:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor, int(row["label"])


def instantiate_backbone(model_name: str, bench_dir: Path, weights: Path):
    """Construct EXACT DeepfakeBench backbone, not timm/torchvision substitutes."""
    training_dir = bench_dir.expanduser().resolve() / "training"
    if not (training_dir / "networks").is_dir():
        raise FileNotFoundError(f"Missing DeepfakeBench training/networks: {training_dir}")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    sys.path.insert(0, str(training_dir))
    try:
        from networks import BACKBONE
        cfg = {"num_classes": 2, "inc": 3, "dropout": False, "mode": "original"}
        if model_name == "xception":
            # Equivalent to DeepfakeBench's XceptionDetector.build_backbone(),
            # with explicit validation of every loaded ImageNet tensor.
            model = BACKBONE["xception"](cfg)
            raw = torch.load(weights, map_location="cpu", weights_only=True)
            if not isinstance(raw, dict) or not raw:
                raise ValueError("Unexpected ImageNet Xception checkpoint format")
            expected = model.state_dict()
            keep = {}
            mismatched = {}
            for key, tensor in raw.items():
                if not torch.is_tensor(tensor):
                    continue
                if "pointwise" in key and tensor.ndim == 2:
                    tensor = tensor.unsqueeze(-1).unsqueeze(-1)
                if key not in expected:
                    continue
                if tensor.shape != expected[key].shape:
                    mismatched[key] = [list(tensor.shape), list(expected[key].shape)]
                    continue
                keep[key] = tensor
            # Classifier is intentionally reinitialized for binary REAL/FAKE.
            backbone_keys = [
                key for key in expected
                if not key.startswith("last_linear.") and not key.startswith("fc.")
            ]
            missing = sorted(set(backbone_keys) - set(keep))
            if missing:
                raise ValueError(
                    "Incomplete ImageNet Xception backbone initialization "
                    f"({len(missing)} missing keys): {missing[:15]}"
                )
            model.load_state_dict(keep, strict=False)
            provenance = {
                "initialization": "ImageNet Xception (DeepfakeBench/Cadene-compatible)",
                "loaded_tensors": len(keep),
                "reinitialized_head": sorted(set(expected) - set(keep)),
                "shape_mismatches": mismatched,
            }
        elif model_name == "efficientnetb4":
            # DeepfakeBench uses efficientnet_pytorch, not torchvision.
            # Restore compatible ImageNet stem after DeepfakeBench reconstruction.
            model = BACKBONE["efficientnetb4"]({**cfg, "pretrained": str(weights)})
            # Upstream DeepfakeBench reconstructs _conv_stem *after* loading
            # ImageNet weights. Restore that exact compatible stem tensor.
            image_net = torch.load(weights, map_location="cpu", weights_only=True)
            stem = image_net.get("_conv_stem.weight") if isinstance(image_net, dict) else None
            target = model.efficientnet._conv_stem.weight
            if not torch.is_tensor(stem) or stem.shape != target.shape:
                raise ValueError("Missing/incompatible EfficientNet-B4 ImageNet stem weights")
            with torch.no_grad():
                target.copy_(stem)
            provenance = {
                "initialization": "efficientnet_pytorch ImageNet-B4 via DeepfakeBench",
                "reinitialized_head": ["last_layer.*"],
                "restored_imagenet_stem": True,
                "note": "Preserves the DeepfakeBench architecture and restores ImageNet stem initialization",
            }
        else:
            raise ValueError(f"Unsupported model {model_name}")
        result = model
    finally:
        sys.path.remove(str(training_dir))
    return result, provenance


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    y_true, y_probs = [], []
    for images, labels in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            logits = model(images.to(device, non_blocking=True))
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise RuntimeError(f"Expected binary detector logits [N,2], got {tuple(logits.shape)}")
        probs = logits.float().softmax(dim=1)[:, 1].cpu().numpy()
        y_probs.extend(float(x) for x in probs)
        y_true.extend(int(x) for x in labels)
    if len(set(y_true)) < 2:
        raise ValueError("Detector holdout must contain REAL and FAKE labels")
    return {
        "auroc": float(roc_auc_score(y_true, y_probs)),
        "accuracy": float(accuracy_score(y_true, np.asarray(y_probs) >= 0.5)),
        "frames": len(y_true),
        "real": int(sum(v == 0 for v in y_true)),
        "fake": int(sum(v == 1 for v in y_true)),
    }


def run(args):
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("selection_metric") != "detector_holdout_auroc":
        raise ValueError("Only detector-holdout AUROC checkpoint selection is supported")
    if not config.get("train_all_layers", False):
        raise ValueError("This protocol requires full-model fine-tuning")
    if config.get("scheduler") != "none" or config.get("optimizer") != "adam":
        raise ValueError("Only the explicitly recorded Adam/no-scheduler protocol is implemented")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    source_path = Path(args.source_manifest)
    source_rows = load_manifest(source_path)
    source_audit = audit(source_rows)
    split_dir = Path(args.manifest_dir)
    report = json.loads((split_dir / "split_report.json").read_text(encoding="utf-8"))
    if report["manifest_sha256"] != sha256(source_path):
        raise ValueError("Source manifest changed since detector split creation")
    if int(report["seed"]) != seed:
        raise ValueError("Seed differs from frozen detector split")
    train_path, holdout_path = (split_dir / "detector_train.csv",
                                split_dir / "detector_holdout.csv")
    train, held = load_manifest(train_path), load_manifest(holdout_path)
    fit = [row for row in source_rows if row["split"] == "fit"]
    expected_keys = {(row["path"], row["group_id"], row["label"]) for row in fit}
    train_keys = {(row["path"], row["group_id"], row["label"]) for row in train}
    hold_keys = {(row["path"], row["group_id"], row["label"]) for row in held}
    if train_keys & hold_keys or train_keys | hold_keys != expected_keys:
        raise ValueError("Detector train/holdout do not exactly partition FIT")
    if {r["group_id"] for r in train} & {r["group_id"] for r in held}:
        raise ValueError("Source-group leakage between detector train and holdout")
    for name, rows in (("train", train), ("holdout", held)):
        if {r["label"] for r in rows} != {"0", "1"}:
            raise ValueError(f"{name} missing one of the two labels")
        if any(r["split"] != "fit" or r["official_split"] != "train" for r in rows):
            raise ValueError(f"{name} contains non-FF++-training frames")

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("Output run directory is not empty; refusing to overwrite checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = int(config["image_size"])
    mean = config["normalization_mean"]
    std = config["normalization_std"]
    training_data = FaceFrames(train, args.image_root, images, mean, std,
                               float(config["horizontal_flip_probability"]))
    holdout_data = FaceFrames(held, args.image_root, images, mean, std)
    batch = int(config["batch_size"])
    accumulate = int(config["gradient_accumulation_steps"])
    if min(batch, accumulate, int(config["epochs"])) < 1:
        raise ValueError("Invalid batch size, accumulation, or epoch count")
    labels = np.asarray([int(r["label"]) for r in train], dtype=int)
    if config["class_balanced_sampling"]:
        counts = np.bincount(labels, minlength=2)
        sampler = WeightedRandomSampler(
            torch.tensor([1.0 / counts[y] for y in labels], dtype=torch.double),
            num_samples=len(train), replacement=True,
            generator=torch.Generator().manual_seed(seed))
        shuffle = False
    else:
        sampler, shuffle = None, True
    loader = DataLoader(training_data, batch_size=batch, sampler=sampler,
                        shuffle=shuffle, num_workers=int(config["num_workers"]),
                        pin_memory=torch.cuda.is_available())
    validation_loader = DataLoader(holdout_data, batch_size=batch, shuffle=False,
                                   num_workers=int(config["num_workers"]),
                                   pin_memory=torch.cuda.is_available())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp = bool(config["use_amp"]) and device.type == "cuda"
    weights_path = Path(args.imagenet_weights).resolve()
    model, initialization = instantiate_backbone(args.model, Path(args.bench_dir), weights_path)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss()
    history, best_auc = [], float("-inf")
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen = 0.0, 0
        for step, (x, y) in enumerate(loader):
            group_start = (step // accumulate) * accumulate
            group_size = min(accumulate, len(loader) - group_start)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss / group_size).backward()
            total_loss += float(loss.item()) * len(y)
            seen += len(y)
            if (step + 1) % accumulate == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(model, validation_loader, device, amp)
        metrics.update(epoch=epoch, train_loss=total_loss / seen)
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["auroc"] > best_auc:
            best_auc = metrics["auroc"]
            torch.save({
                "state_dict": {k: tensor.detach().cpu() for k, tensor in model.state_dict().items()},
                "model": args.model,
                "epoch": epoch,
                "holdout_metrics": metrics,
                "imagenet_sha256": sha256(weights_path),
                "source_manifest_sha256": sha256(source_path),
                "detector_train_manifest_sha256": sha256(train_path),
                "detector_holdout_manifest_sha256": sha256(holdout_path),
                "config_sha256": sha256(config_path),
                "seed": seed,
                "initialization": initialization,
            }, out_dir / f"{args.model}_best.pth")

    report = {
        "model": args.model,
        "model_library": "DeepfakeBench BACKBONE",
        "upstream_backbone": "efficientnet_pytorch" if args.model == "efficientnetb4" else "DeepfakeBench/FF++ Xception",
        "imagenet_weights_sha256": sha256(weights_path),
        "source_manifest_sha256": sha256(source_path),
        "detector_train_manifest_sha256": sha256(train_path),
        "detector_holdout_manifest_sha256": sha256(holdout_path),
        "config": config,
        "config_sha256": sha256(config_path),
        "source_manifest_audit": source_audit,
        "detector_train_frames": len(train),
        "detector_holdout_frames": len(held),
        "detector_train_groups": len({r["group_id"] for r in train}),
        "detector_holdout_groups": len({r["group_id"] for r in held}),
        "train_label_counts": dict(Counter(r["label"] for r in train)),
        "holdout_label_counts": dict(Counter(r["label"] for r in held)),
        "initialization": initialization,
        "selection_metric": "detector_holdout_auroc",
        "best_epoch": max(history, key=lambda s: s["auroc"])["epoch"],
        "best_auroc": best_auc,
        "history": history,
        "scope": "Detector trained only on a subset of TRIAGE FIT groups; no CAL/VAL/TEST",
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"BEST CHECKPOINT: {out_dir / (args.model + '_best.pth')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["xception", "efficientnetb4"])
    p.add_argument("--config", required=True)
    p.add_argument("--source-manifest", required=True,
                   help="Complete original FF++ FIT/CAL/VAL/TEST manifest; audited for group leakage")
    p.add_argument("--manifest-dir", required=True,
                   help="Frozen detector_train.csv, detector_holdout.csv, split_report.json")
    p.add_argument("--image-root", default=".")
    p.add_argument("--bench-dir", required=True)
    p.add_argument("--imagenet-weights", required=True,
                   help="Verified ImageNet initialization checkpoint, NOT an FF++ detector")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
