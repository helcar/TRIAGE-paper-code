"""Unit tests for the leakage-critical, dependency-light manifest logic."""
import csv
from pathlib import Path

import pytest

from triage_repro.manifest import audit, load_manifest, write_rows
from triage_repro.make_detector_split import partition


def rows():
    items = []
    # Source groups deliberately have multiple video/method instances.
    for i in range(20):
        split = "fit" if i < 12 else ("cal" if i < 15 else ("val" if i < 18 else "test"))
        official = "train" if split in {"fit", "cal"} else split
        for label, method in (("0", "original"), ("1", "Deepfakes")):
            for frame in range(2):
                items.append({
                    "path": f"/synthetic/{i}/{method}/{frame}.png",
                    "label": label,
                    "split": split,
                    "group_id": str(i),
                    "video_id": f"clip{i}",
                    "method": method,
                    "official_split": official,
                })
    return items


def test_audit_and_deterministic_detector_split():
    source = rows()
    assert audit(source)["partition_frames"]["fit"] == 48
    train_a, held_a = partition(source, seed=42, holdout_fraction=0.25)
    train_b, held_b = partition(source, seed=42, holdout_fraction=0.25)
    assert train_a == train_b and held_a == held_b
    assert len(train_a) + len(held_a) == 48
    assert not ({r["group_id"] for r in train_a} & {r["group_id"] for r in held_a})
    assert {r["label"] for r in train_a} == {"0", "1"}


def test_reject_group_leakage():
    source = rows()
    source[-1]["group_id"] = source[0]["group_id"]
    with pytest.raises(ValueError, match="overlap"):
        audit(source)


def test_reject_test_inside_fit():
    source = rows()
    source[0]["official_split"] = "test"
    with pytest.raises(ValueError, match="mismatch"):
        audit(source)


def test_reject_more_than_two_frames():
    source = rows()
    row = dict(source[0], path="/synthetic/0/original/extra.png")
    source.append(row)
    with pytest.raises(ValueError, match="More than 2"):
        audit(source)


def test_csv_roundtrip(tmp_path: Path):
    source = rows()
    file = tmp_path / "manifest.csv"
    write_rows(file, source)
    assert load_manifest(file) == source


def test_reject_bad_holdout():
    with pytest.raises(ValueError, match="holdout_fraction"):
        partition(rows(), seed=42, holdout_fraction=0.99)
