"""Create a source-safe FF++ c23 manifest from already aligned face crops.

Crops layout:
  <crop-root>/original_sequences/youtube/c23/<images|frames>/<video_id>/*.png
  <crop-root>/manipulated_sequences/<method>/c23/<images|frames>/<a_b>/*.png
Or supply --real-dir and repeated --fake-dir METHOD=PATH to override.
Use official train.json/val.json/test.json lists of original-ID pairs,
available in the FaceForensics repository dataset/splits directory.
No face detection is performed here: use the checkpoint-matched preprocessing.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

SPLITS = ("fit", "cal", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png"}


class UF:
    def __init__(self): self.p = {}
    def find(self, k):
        self.p.setdefault(k, k)
        if self.p[k] != k: self.p[k] = self.find(self.p[k])
        return self.p[k]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[max(ra, rb)] = min(ra, rb)


def load_official_splits(directory):
    """Map original-video IDs to official split; check video-ID overlap."""
    result = {}
    for split, filename in (("train", "train.json"), ("val", "val.json"),
                            ("test", "test.json")):
        file = Path(directory) / filename
        with file.open(encoding="utf-8") as f:
            pairs = json.load(f)
        if not isinstance(pairs, list):
            raise ValueError(f"{file} must contain a list of ID pairs")
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Bad source pair {pair!r} in {file}")
            for src in pair:
                src = str(src)
                old = result.setdefault(src, split)
                if old != split:
                    raise ValueError(f"Official split leakage for {src}: {old}, {split}")
    return result


def indexed_videos(directory):
    base = Path(directory)
    if not base.is_dir():
        raise FileNotFoundError(base)
    items = {}
    for folder in sorted(base.iterdir()):
        if folder.is_dir():
            frames = sorted(p for p in folder.iterdir()
                            if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if frames:
                items[folder.name] = frames
    if not items:
        raise ValueError(f"No <video-id>/<frame>.jpg or .png folders in {base}")
    return items


def sample(frames, per_video):
    n = len(frames)
    count = min(per_video, n)
    return [frames[min(n-1, int((i+0.5)*n/count))] for i in range(count)]


def prepare(real, fakes, official_dir, output, per_video=8, cal_fraction=.2, seed=42):
    official = load_official_splits(official_dir)
    videos = [(0, "original", indexed_videos(real))]
    videos += [(1, method, indexed_videos(folder)) for method, folder in fakes]
    uf = UF()
    for _, _, items in videos:
        for name in items:
            parts = name.split("_")
            if len(parts) == 1:
                uf.find(parts[0])
            elif len(parts) == 2:
                uf.union(*parts)
            else:
                raise ValueError(f"Unexpected video ID {name!r}")
    components = defaultdict(set)
    for node in list(uf.p):
        components[uf.find(node)].add(node)
    group_official = {}
    for group, members in components.items():
        labels = {official[m] for m in members if m in official}
        if len(labels) > 1:
            raise ValueError(f"Cross-split connected component {group}: {members}")
        if len(labels) == 1:
            group_official[group] = labels.pop()
    train_groups = sorted(g for g,s in group_official.items() if s == "train")
    rng = np.random.default_rng(seed)
    rng.shuffle(train_groups)
    ncal = max(1, round(len(train_groups)*cal_fraction))
    calibration = set(train_groups[:ncal])
    mapping = {"train": "fit", "val": "val", "test": "test"}
    rows, report = [], defaultdict(int)
    skipped = set()
    for label, method, items in videos:
        for name, frames in items.items():
            parts = name.split("_")
            # Fail closed: no unknown source video or manipulated pair may leak in.
            if any(p not in official for p in parts):
                skipped.add(name)
                continue
            group = uf.find(parts[0])
            official_split = group_official.get(group)
            if official_split is None:
                skipped.add(name)
                continue
            split = ("cal" if group in calibration else "fit") if official_split=="train" else mapping[official_split]
            for frame in sample(frames, per_video):
                rows.append({"path":str(frame.resolve()), "label":str(label),
                             "split":split, "group_id":group,
                             "video_id":name, "method":method,
                             "official_split":official_split})
                report[split] += 1
    if not rows:
        raise ValueError("No matching frame crops; check crop directories and split metadata")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "path", "label", "split", "group_id", "video_id",
            "method", "official_split"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"manifest":str(output), "frames":len(rows),
                      "counts":dict(report), "source_groups":len(group_official),
                      "skipped_videos":len(skipped)}, indent=2))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-dir", required=True)
    p.add_argument("--fake-dir", action="append", required=True,
                   help="METHOD=PATH; repeat per manipulation type")
    p.add_argument("--splits-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frames-per-video", type=int, default=8)
    p.add_argument("--cal-fraction", type=float, default=.2)
    p.add_argument("--seed", type=int, default=42)
    args=p.parse_args()
    if not 0 < args.cal_fraction < 1 or args.frames_per_video < 1:
        p.error("cal-fraction must lie in (0,1), and frames-per-video must be positive")
    fakes=[]
    for item in args.fake_dir:
        if "=" not in item: p.error("--fake-dir must be METHOD=PATH")
        method,path=item.split("=",1)
        if not method or not path: p.error("--fake-dir needs nonempty METHOD and PATH")
        fakes.append((method,path))
    prepare(args.real_dir,fakes,args.splits_dir,args.output,
            args.frames_per_video,args.cal_fraction,args.seed)


if __name__=="__main__":
    main()
