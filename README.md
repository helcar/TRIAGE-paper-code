# TRIAGE: Explainable Error Ranking for Face Deepfake Detectors

**Reproducibility source for the IEEE TIFS TRIAGE manuscript.** TRIAGE scores the risk that an already-frozen deepfake detector has made a mistake. It learns detector-local visual prototypes, separates REAL- and FAKE-supporting signed gradient × activation evidence, learns NMF attribution topics, and compares predictions against multiple correct-prediction reference profiles.

**Release status:** The research-method code and new unified workflow are available here. This repository is currently **private**. The newly proposed ImageNet-initialized CNN training procedure and its downstream TRIAGE results have **not** yet been executed and independently validated. Existing v14 paper numbers are from an earlier experiment history and must not be attributed to the new detector checkpoints until rerun. No raw datasets or third-party model weights are committed.

## One supported workflow

| Stage | Authoritative module | Main rule |
| --- | --- | --- |
| Dataset preparation | `code.prepare_ffpp_c23_manifest`, `code.prepare_external_video_manifest`, `code.preprocess_ffpp_dlib`, `code.preprocess_external_videos` | Source-group-aware FIT/CAL/VAL/TEST manifests; at most two face crops/video |
| Detector split | `triage_repro.make_detector_split` | Split FF++ FIT into detector train/holdout by source group |
| CNN training | `triage_repro.train_cnn` | DeepfakeBench architectures + ImageNet weights; no FF++ pretrained detector weights |
| ViT training | `code.train_ffpp_vit_b7` | CommunityForensics init + FF++ FIT only |
| Attribution extraction | `code.extract_ffpp_large`, `code.extract_efficientnetb4`, `code.extract_community_vit_b7` | Frozen detector; same layer for all partitions of an experiment |
| Layer / hyperparameter selection | `triage_repro.select_layer`, `code.search_triage_sensitivity` | FIT prototypes/topics, CAL reference profiles, VAL selection; never select on TEST |
| Final evaluation | `triage_repro.evaluate_final` | Replay VAL first, then six ranking signals on TEST |
| Supplementary analyses | `code.evaluate_ffpp_reference_lomo`, `code.topic_sensitivity_v11`, `code.report_practical_results` and related modules | Use the corresponding frozen detector and source-group-safe manifests |

The convenience CLI `triage_repro.workflow` orchestrates training, layer screening, validation selection, and final test evaluation from a single YAML file. It supports all nine combinations of **Xception / EfficientNet-B4 / ViT × FF++ c23 / Celeb-DF-v2 / DFDC**.

## Method implementation

The final method is in [`code/triage.py`](code/triage.py). Its inputs are the fixed detector's local layer activations and signed gradient × activation, not ground-truth test labels. The core processes are:

1. **FIT:** Normalize local feature vectors; use MiniBatchKMeans to learn `K` detector-local prototypes. Aggregate positive and negative signed attributions per prototype. Learn one NMF topic basis per polarity.
2. **CAL:** Project each calibration example into topic distributions. Build `J` predicted-class-specific profiles **only from correctly classified calibration examples**.
3. **VAL:** Select the detector layer and `K, M, J, tau`, and score-term weights by error-ranking AUROC (AUPRC tie-breaker). Neither the detector nor TRIAGE is selected using TEST.
4. **TEST:** Calculate class-conditional profile discrepancy and per-topic risk explanations for each prediction. Compare against confidence, centroid, Mahalanobis, kNN, and prototype-attribution divergence. Report AUROC, AUPRC, risk-coverage and review-budget metrics, with source-group bootstrap uncertainty.

Detector features and gradients are extracted from actual DeepfakeBench blocks: Xception `backbone.block8` in the v14 paper; EfficientNet-B4 `backbone.efficientnet._blocks.29` or `_blocks.15` depending on dataset. New CNN retraining may change the validation-selected block; the workflow screens candidate layers again.

## Environment

Python 3.10+ and PyTorch compatible with your CUDA installation; install this repository's dependencies (see `requirements-cnn.txt`) and the corresponding **separately cloned DeepfakeBench** dependencies. ViT also needs `transformers`, `huggingface_hub` and `safetensors`. Face alignment requires DeepfakeBench's dlib landmark assets and licenses.

DeepfakeBench's own ImageNet initialization files are **not** FF++-trained detector checkpoints. Fetch them from its [v1.0.0 ImageNet initialization archive](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip) using:

```bash
python -m triage_repro.fetch_imagenet --model xception --out-dir checkpoints
python -m triage_repro.fetch_imagenet --model efficientnetb4 --out-dir checkpoints
```

The downloader records hashes and never commits or redistributes weights.

Copy `configs/workflow.example.yaml`, replace the dataset paths, DeepfakeBench path, checkpoint paths, and desired output locations, and then use:

```bash
# A GPU-free preview of the validation-only workflow
python -m triage_repro.workflow --config configs/workflow.example.yaml --experiment xception_ffpp --stage select --dry-run

# When the private dataset/weights are ready, stages are: audit, split,
# train, select (layer + hyperparameters), test (VAL replay + TEST evaluation).
```

Run `split` and `train` only for the FF++ experiment of each detector. For Celeb-DF-v2 and DFDC, use that detector's same frozen FF++ fine-tuned checkpoint but fit/calibrate TRIAGE on each corresponding target dataset. For the exact expected artifact names and model-specific considerations, see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Expected outputs

Under ignored local `runs/` directories the workflow produces detector training reports/checkpoints, frozen detector feature/attribution arrays, per-layer validation search candidates and choices, final detector-error metrics, per-frame six-signal scores, topic explanations, and a serialized fitted TRIAGE model. All arrays and score files must be regenerated from the corresponding checkpoint and manifest, not mixed between experiment versions.

The original manuscript's v14-selected settings are preserved as a **historical reference only** in [`configs/paper_v14_reference.yaml`](configs/paper_v14_reference.yaml). They must not be substituted for new validation-selected settings after the CNN checkpoints change.

## Privacy, provenance, and limitations

The repository excludes pretrained weights, raw or aligned face images, videos, non-sanitized split manifests, and large arrays. Celeb-DF-v2's group split does **not** guarantee celebrity identity separation. The DFDC evaluation uses a two-archive subset of the public **training** dataset, not the official challenge test. CNN and ViT fine-tuning currently overlap the TRIAGE FIT partition by design, but not TRIAGE CAL/VAL/TEST. For complete detector-versus-TRIAGE-FIT independence, the detector would need separately reserved FF++ source groups, requiring all experiments to be repeated.

Before converting this repository to public, select a license for author-owned source, verify third-party terms and dataset licenses, sanitize all generated files, and regenerate the paper tables with the controlled CNN checkpoints. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and [`docs/CNN_PROTOCOL.md`](docs/CNN_PROTOCOL.md).
