"""Run DeepfakeBench dlib alignment on FF++ c23 videos without optional masks.

This imports DeepfakeBench's extract_aligned_face_dlib implementation instead
of reimplementing its facial landmark geometry. Output matches its frames/
layout, next to original c23 videos. Source videos are never modified.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import dlib
import numpy as np


CATEGORIES = {
    "original": Path("original_sequences/youtube/c23"),
    "Deepfakes": Path("manipulated_sequences/Deepfakes/c23"),
    "Face2Face": Path("manipulated_sequences/Face2Face/c23"),
    "FaceSwap": Path("manipulated_sequences/FaceSwap/c23"),
    "NeuralTextures": Path("manipulated_sequences/NeuralTextures/c23"),
}


def load_bench_alignment(bench_dir):
    source = Path(bench_dir).resolve() / "preprocessing" / "preprocess.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("deepfakebench_face_preprocess", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_aligned_face_dlib


def indices(length, count):
    if length <= 0:
        return []
    return sorted(set(int(i) for i in np.linspace(0, length - 1,
                                                   min(length, count), dtype=int)))


def run(root, bench_dir, categories, count, max_videos, overwrite=False):
    root = Path(root)
    predictor_path = (Path(bench_dir).resolve() / "preprocessing" / "dlib_tools"
                      / "shape_predictor_81_face_landmarks.dat")
    align = load_bench_alignment(bench_dir)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))
    stats = {"videos": 0, "saved": 0, "no_face": 0, "failed_reads": 0,
             "skipped_completed": 0, "videos_with_no_crops": 0}
    for category in categories:
        base = root / CATEGORIES[category]
        files = sorted((base / "videos").glob("*.mp4"))
        if not files:
            raise FileNotFoundError(f"No videos in {base / 'videos'}")
        if max_videos is not None:
            files = files[:max_videos]
        for path in files:
            output = base / "frames" / path.stem
            marker = output / "_complete.json"
            if marker.exists() and not overwrite:
                stats["skipped_completed"] += 1
                continue
            output.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                stats["failed_reads"] += 1
                continue
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            saved_this = 0
            attempted = 0
            for index in indices(n, count):
                dst = output / f"{index:03d}.png"
                if dst.exists() and not overwrite:
                    saved_this += 1
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                attempted += 1
                if not ok:
                    stats["failed_reads"] += 1
                    continue
                face, _, _ = align(detector, predictor, frame, res=256)
                if face is None:
                    stats["no_face"] += 1
                    continue
                if not cv2.imwrite(str(dst), face):
                    raise OSError(f"Cannot write {dst}")
                saved_this += 1
            cap.release()
            stats["saved"] += saved_this
            stats["videos"] += 1
            if saved_this == 0:
                stats["videos_with_no_crops"] += 1
            marker.write_text(json.dumps({"source": path.name,
                                          "requested_frames":count,
                                          "decoded_total":n,
                                          "saved_frames":saved_this,
                                          "attempted":attempted}))
            print(json.dumps({"category":category,"video":path.name,
                              "saved":saved_this,"requested":count}),flush=True)
    print(json.dumps({"summary":stats},indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--categories", nargs="+", choices=list(CATEGORIES),
                    default=["original"])
    ap.add_argument("--frames-per-video", type=int, default=8)
    ap.add_argument("--max-videos", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.frames_per_video < 1 or (args.max_videos is not None and args.max_videos < 1):
        ap.error("frame count and max-videos must be positive")
    run(args.root,args.bench_dir,args.categories,
        args.frames_per_video,args.max_videos,args.overwrite)


if __name__ == "__main__":
    main()
