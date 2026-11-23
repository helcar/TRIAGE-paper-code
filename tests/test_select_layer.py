"""Synthetic validation-layer screen tests (no datasets or model weights)."""
import json
from pathlib import Path

import numpy as np
import pytest

from triage_repro.select_layer import choose


def candidate(tmp_path, name, score, checkpoint="checkpoint-a", logits=None):
    root = tmp_path / name
    root.mkdir()
    for split in ("fit", "cal", "val"):
        folder = root / split
        folder.mkdir()
        marker = {
            "layer": name, "checkpoint_sha256": checkpoint,
            "manifest_sha256": "manifest-sha",
            "image_size": 256, "normalization": "[-1,1]",
        }
        (folder / "_complete.json").write_text(json.dumps(marker))
    np.save(root / "val" / "logits.npy",
            np.asarray(logits if logits is not None else [0.5, -0.4, 0.1], np.float32))
    np.save(root / "val" / "labels.npy", np.asarray([1, 0, 1], np.int64))
    grid = root / "grid.json"
    grid.write_text(json.dumps({
        "search": {"seed": 42, "K": [64], "M": [4], "J": [1]},
        "best": {"full_topic": {
            "kind": "full_topic", "auroc": score, "auprc": 0.4,
        }},
    }))
    return (name, root, grid)


def test_choose_best_validation_layer(tmp_path):
    a = candidate(tmp_path, "layer4", 0.65)
    b = candidate(tmp_path, "layer8", 0.85)
    report = choose([a, b], tmp_path / "chosen.json")
    assert report["selected_layer"] == "layer8"
    assert report["test_accessed"] is False
    assert len(report["candidates"]) == 2


def test_different_detectors_cannot_be_compared(tmp_path):
    a = candidate(tmp_path, "layer4", 0.65)
    b = candidate(tmp_path, "layer8", 0.85, checkpoint="checkpoint-b")
    with pytest.raises(ValueError, match="different detector"):
        choose([a, b], tmp_path / "chosen.json")


def test_val_predictions_must_match(tmp_path):
    a = candidate(tmp_path, "layer4", 0.65)
    b = candidate(tmp_path, "layer8", 0.85, logits=[0.6, -0.2, 0.3])
    with pytest.raises(ValueError, match="logits"):
        choose([a, b], tmp_path / "chosen.json")


def test_selection_is_immutable(tmp_path):
    a = candidate(tmp_path, "layer4", 0.65)
    b = candidate(tmp_path, "layer8", 0.85)
    out = tmp_path / "chosen.json"
    choose([a, b], out)
    with pytest.raises(FileExistsError):
        choose([a, b], out)
