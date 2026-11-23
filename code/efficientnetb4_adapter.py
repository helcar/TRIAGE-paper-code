"""Strict DeepfakeBench EfficientNet-B4 adapter for a FROZEN deepfake detector.

ImageNet initialization in training/pretrained is NOT a trained deepfake
detector. Pass a complete deepfake-trained checkpoint, usually from
training/weights, with the two-class classifier included.

Uses DeepfakeBench's efficientnet_pytorch backbone, not torchvision's B4.
Constructs it WITHOUT ImageNet weights, then strictly loads the trained
detector, including its classifier. No partial/random head is acceptable.
"""
from pathlib import Path
import sys

from .xception_adapter import BinaryLogitAdapter, load_detector_weights, resolve_layer


def build_deepfakebench_efficientnetb4(bench_dir, checkpoint, device="cuda"):
    bench = Path(bench_dir).expanduser().resolve()
    training = bench / "training"
    if not (training / "networks").is_dir():
        raise FileNotFoundError(
            f"DeepfakeBench training/networks missing: {training}")
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Deepfake-trained EfficientNet-B4 checkpoint not found: {path}\n"
            "The file training/pretrained/efficientnet-b4-6ed6700e.pth "
            "is only ImageNet initialization. Find the corresponding "
            "deepfake-trained detector checkpoint, usually in training/weights.")
    if "pretrained" in [part.lower() for part in path.parts]:
        raise ValueError(
            "Refusing weights from training/pretrained: they are typically "
            "ImageNet initialization, not a trained deepfake detector. "
            "Use a documented deepfake-trained full checkpoint.")
    sys.path.insert(0, str(training))
    try:
        try:
            from networks import BACKBONE
        except ModuleNotFoundError as exc:
            if exc.name == "efficientnet_pytorch":
                raise ModuleNotFoundError(
                    "DeepfakeBench EfficientNet-B4 requires efficientnet_pytorch. "
                    "Install it in your .venv with "
                    "'python -m pip install efficientnet_pytorch'.") from exc
            raise
        # DeepfakeBench Registry implements __getitem__ but not
        # __contains__ or __iter__. Using 'x in BACKBONE' tries integer
        # indexing starting at 0 and fails with KeyError: 0.
        try:
            backbone_class = BACKBONE["efficientnetb4"]
        except KeyError as exc:
            available = sorted(getattr(BACKBONE, "data", {}))
            raise RuntimeError(
                "DeepfakeBench did not register 'efficientnetb4'; "
                f"available backbones: {available}. "
                "Check training/networks/__init__.py."
            ) from exc
        backbone = backbone_class({
            "num_classes": 2, "inc": 3,
            "dropout": False, "mode": "original",
            "pretrained": False,
        })
        info = load_detector_weights(backbone, path)
    except RuntimeError as exc:
        if "No complete Xception backbone match" in str(exc):
            raise RuntimeError(
                "EfficientNet-B4 checkpoint did not match the exact "
                "DeepfakeBench backbone and TWO-CLASS detector head. "
                "Do not use partial loading or ImageNet-only weights; "
                "the checkpoint might be from another architecture version.\n"
                + str(exc)) from exc
        raise
    finally:
        sys.path.remove(str(training))
    model = BinaryLogitAdapter(backbone).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    info["architecture"] = "DeepfakeBench EfficientNetB4"
    info["checkpoint"] = str(path)
    info["layer_naming_example"] = "backbone.efficientnet._blocks.21"
    return model, info
