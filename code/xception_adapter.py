"""DeepfakeBench Xception adapter for TRIAGE.

Requires an independent checkout of SCLBD/DeepfakeBench and a *deepfake-trained*
checkpoint. The ImageNet Xception initialization weights are NOT a detector.
"""
from pathlib import Path
import sys
import torch
from torch import nn


class BinaryLogitAdapter(nn.Module):
    """Convert a two-logit classifier to the pre-sigmoid fake-vs-real log-odds.

    For a softmax classifier, logit_fake - logit_real is the exact binary
    decision logit, preserving its probability and gradient direction.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        features = self.backbone.features(images)
        logits = self.backbone.classifier(features)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"Expected [B,2] logits, got {tuple(logits.shape)}")
        return logits[:, 1] - logits[:, 0]


def resolve_layer(model, path):
    """Resolve a named nested module, e.g. 'backbone.block12'."""
    modules = dict(model.named_modules())
    if path not in modules:
        nearby = [x for x in modules if path.lower() in x.lower()]
        raise ValueError(f"Layer {path!r} not found. Similar: {nearby[:12]}")
    return modules[path]


def _unpack_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    for key in ("state_dict", "model_state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    if not checkpoint or not all(torch.is_tensor(v) for v in checkpoint.values()):
        raise ValueError("No tensor state_dict found in checkpoint")
    return checkpoint


def _candidate_mappings(state):
    yield state
    for prefix in ("module.", "backbone.", "module.backbone.", "model.", "model.backbone."):
        stripped = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if stripped:
            yield stripped


def load_detector_weights(backbone, checkpoint_path):
    """Load an exact matching detector checkpoint; refuse partial/random weights.

    Only load checkpoints from trusted sources. This uses weights_only=True.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = _unpack_state(torch.load(path, map_location="cpu", weights_only=True))
    expected = backbone.state_dict()
    errors = []
    for candidate in _candidate_mappings(state):
        missing = set(expected) - set(candidate)
        mismatched = [k for k in expected.keys() & candidate.keys()
                      if expected[k].shape != candidate[k].shape]
        unexpected = set(candidate) - set(expected)
        if not missing and not mismatched:
            # Extra keys are allowed for full detector checkpoints, whose
            # classifier/loss/other modules might share the file.
            backbone.load_state_dict({k: candidate[k] for k in expected}, strict=True)
            return {"loaded_tensors": len(expected), "ignored_tensors": len(unexpected)}
        errors.append((len(missing), len(mismatched)))
    raise RuntimeError(
        "No complete Xception backbone match in checkpoint. "
        "Possible wrong architecture, ImageNet-only weights, or wrapper key prefix. "
        f"(missing, mismatched) across candidates: {errors}")


def build_deepfakebench_xception(bench_dir, checkpoint, device="cuda"):
    """Build the official DeepfakeBench Xception backbone without its trainer."""
    training = Path(bench_dir).expanduser().resolve() / "training"
    if not (training / "networks").is_dir():
        raise FileNotFoundError(f"DeepfakeBench training/networks missing: {training}")
    sys.path.insert(0, str(training))
    try:
        from networks import BACKBONE
        # DeepfakeBench registers the Xception backbone under 'xception'.
        try:
            xception_class = BACKBONE["xception"]
        except KeyError as exc:
            raise RuntimeError("DeepfakeBench BACKBONE registry does not contain xception") from exc
        backbone = xception_class({
            "mode": "original", "num_classes": 2, "inc": 3, "dropout": False})
        info = load_detector_weights(backbone, checkpoint)
    finally:
        sys.path.remove(str(training))
    model = BinaryLogitAdapter(backbone).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, info
