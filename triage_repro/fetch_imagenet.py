"""Fetch original ImageNet initialization weights from DeepfakeBench release.

Downloads only on explicit invocation, checks expected archive members and
records checksums; NEVER downloads DeepfakeBench's trained FF++ detector weights.
Weights are not redistributed by TRIAGE-paper-code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

BUNDLE_URL = "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip"
ALLOWED = {
    "xception": "xception-b5690688.pth",
    "efficientnetb4": "efficientnet-b4-6ed6700e.pth",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=sorted(ALLOWED), required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--archive", help="Optional previously downloaded DeepfakeBench pretrained.zip")
    args = p.parse_args()
    target_dir = Path(args.out_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = ALLOWED[args.model]
    destination = target_dir / filename
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing initialization file: {destination}")
    archive = Path(args.archive).resolve() if args.archive else target_dir / "pretrained.zip"
    if not archive.is_file():
        if args.archive:
            raise FileNotFoundError(archive)
        print(f"Downloading original ImageNet initialization bundle: {BUNDLE_URL}", flush=True)
        temp = archive.with_suffix(".partial")
        if temp.exists():
            raise FileExistsError(f"Remove incomplete partial file first: {temp}")
        try:
            request = urllib.request.Request(BUNDLE_URL, headers={"User-Agent": "TRIAGE-repro/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as dst:
                shutil.copyfileobj(response, dst)
            temp.replace(archive)
        finally:
            if temp.exists():
                temp.unlink()
    with zipfile.ZipFile(archive, "r") as zf:
        matches = [entry for entry in zf.infolist()
                   if not entry.is_dir() and Path(entry.filename).name == filename]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {filename} in archive; found {len(matches)}")
        entry = matches[0]
        if entry.file_size < 1_000_000 or entry.file_size > 2_000_000_000:
            raise ValueError(f"Unexpected model checkpoint size: {entry.file_size}")
        with zf.open(entry) as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    result = {
        "model": args.model,
        "pretraining": "ImageNet (DeepfakeBench pretrained.zip, NOT FF++-trained weights)",
        "source_url": BUNDLE_URL,
        "archive_sha256": sha256(archive),
        "extracted_file": filename,
        "checkpoint_sha256": sha256(destination),
        "checkpoint_bytes": destination.stat().st_size,
    }
    provenance = destination.with_suffix(".provenance.json")
    provenance.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
