"""Read and audit source-group-aware FF++ frame manifests."""
from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED = {"path", "label", "split", "group_id", "video_id", "method", "official_split"}
SPLITS = ("fit", "cal", "val", "test")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for buf in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"Missing manifest columns: {sorted(REQUIRED - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def audit(rows: list[dict[str, str]]) -> dict:
    groups: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    video_frame_counts = Counter()
    seen = set()
    for row in rows:
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"Unknown TRIAGE split: {split}")
        if row["label"] not in {"0", "1"}:
            raise ValueError("Labels must be 0/1, with 1=FAKE")
        if not row["group_id"] or not row["video_id"] or not row["path"]:
            raise ValueError("Missing source-group, video, or image path")
        key = row["path"].replace("\\", "/").lower()
        if key in seen:
            raise ValueError(f"Repeated image path: {row['path']}")
        seen.add(key)
        # The official FF++ training split is subdivided ONLY into FIT and CAL.
        expected = "train" if split in ("fit", "cal") else split
        if row["official_split"] != expected:
            raise ValueError(f"Official FF++ split mismatch at {row['path']}: {split}/{row['official_split']}")
        groups[row["group_id"]].add(split)
        counts[(split, row["label"])] += 1
        video_frame_counts[(split, row["video_id"], row["method"])] += 1
    leaking = {g: sorted(ss) for g, ss in groups.items() if len(ss) > 1}
    if leaking:
        raise ValueError(f"Source-group overlap across TRIAGE partitions: {dict(list(leaking.items())[:5])}")
    # The experiment uses at most two frames per underlying video/method pair.
    too_many = {str(k): v for k, v in video_frame_counts.items() if v > 2}
    if too_many:
        raise ValueError(f"More than 2 frames for video/method: {dict(list(too_many.items())[:5])}")
    return {
        "frames": len(rows),
        "groups": len(groups),
        "partition_frames": {s: sum(counts[(s, label)] for label in ("0", "1")) for s in SPLITS},
        "partition_real_fake": {s: {"real": counts[(s, "0")], "fake": counts[(s, "1")]} for s in SPLITS},
        "partition_groups": {s: len({r["group_id"] for r in rows if r["split"] == s}) for s in SPLITS},
        "max_frames_per_video": max(video_frame_counts.values()),
    }


def write_rows(path: str | Path, rows: list[dict[str, str]]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
