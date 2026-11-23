# TRIAGE: Explainable Error Ranking for Face Deepfake Detectors

Reproducibility code for the TRIAGE research project (work in progress).

## Status

This repository is being rebuilt from the research prototype. **The ImageNet-initialized CNN retraining experiments have not yet been run**, and their results must not be attributed to the current manuscript until checkpoint hashes, split audits, and all downstream TRIAGE evaluations have been regenerated.

## Detector training protocol

- Architectures: DeepfakeBench `xception` and `efficientnetb4` (based on `efficientnet_pytorch`), retaining the intermediate blocks used by TRIAGE.
- Initial weights: ImageNet initialization configured in DeepfakeBench (`xception-b5690688.pth` and `efficientnet-b4-6ed6700e.pth`), **not** FF++-trained DeepfakeBench release checkpoints.
- Dataset: the audited FF++ c23 TRIAGE frame manifest. A source-group-disjoint detector-training/holdout split is created **within the FIT partition**. CAL, VAL, and TEST are never used to train or select the detector.
- The detector training subset intentionally overlaps TRIAGE FIT; it does not overlap TRIAGE CAL, VAL, or TEST. This is not a fully independent five-way detector/TRIAGE split.
- Select best detector checkpoint using detector-holdout AUROC only; freeze it for TRIAGE evaluation.
- Default training protocol is recorded in `configs/cnn_finetune.yaml`. Defaults are **proposals**, not recovered historical settings.

## Quick start

1. Clone the compatible [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) source and obtain its original ImageNet initialization weights. No model checkpoints, face images, or restricted dataset files are redistributed here.
2. Export your existing FF++ frame manifest with columns `path,label,split,group_id,video_id,method,official_split`. Use the official group-aware partitions.
3. Run `python -m triage_repro.audit_manifest --manifest PATH` to validate split/group independence.
4. Run `python -m triage_repro.make_detector_split --manifest PATH --out-dir OUTPUT` to write frozen detector train/holdout manifests.
5. Run `python -m triage_repro.train_cnn --config configs/cnn_finetune.yaml --model xception --source-manifest FFPP_MANIFEST.csv --manifest-dir OUTPUT --bench-dir PATH_TO_DEEPFAKEBENCH --imagenet-weights PATH_TO_IMAGENET_XCEPTION --out-dir RUN_DIR`. Repeat with `--model efficientnetb4` and the corresponding ImageNet weights and a different run directory.
6. The runner saves a full detector `state_dict` in `xception_best.pth` / `efficientnetb4_best.pth` plus `training_report.json`. These outputs are local and intentionally Git-ignored. Use them with the migrated `code/xception_adapter.py` and `code/efficientnetb4_adapter.py` for frozen TRIAGE feature extraction.

## Included research code

`code/triage.py`, the CNN adapters, extraction scripts, source-group manifest builders, and selected evaluation scripts are migrated from the working research repository. They are **not yet a fully validated end-to-end reproduction release**. The new detector trainer is in `triage_repro/` and its training settings must not be confused with the historical detector checkpoints.


The training runner does not use TRIAGE validation/test labels or the published DFDC/Celeb evaluation sets. Report exactly which manifest and weight SHA-256 hashes generated a checkpoint. Never substitute a different library's ImageNet weights under a matching model name without verifying parameter compatibility.

## Publication preparation

Before public release: run tests, train both CNNs, independently audit detector splits, rerun ALL CNN TRIAGE results, verify checkpoint loading in the frozen TRIAGE adapters, sanitize absolute machine-specific paths in any exported manifests or logs, provide provenance records, and review dataset licensing. Never publish raw FF++/Celeb-DF/DFDC face frames without permission. The repository is currently private; do not cite it as publicly accessible yet.
