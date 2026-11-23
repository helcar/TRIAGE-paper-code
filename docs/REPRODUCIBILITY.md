# Reproducing the final TRIAGE method

This document describes the **authoritative pipeline** in this repository. It supersedes the historical starter instructions from the original research project. The final paper is specified by `main-v14.tex` and `supp-v14.tex` in `Qi---TKDE2026`; their historical settings are copied to `configs/paper_v14_reference.yaml` for auditing only.

## 0. Scope

This code implements frame-level error ranking for frozen binary face-deepfake detectors. It is white-box and needs intermediate features and gradients. At test time it uses detector outputs and previously learned reference profiles, **not test ground truth**. Labels are used only when calculating evaluation metrics.

Evaluation datasets:
- FF++ c23: official test source groups and held-out official validation groups
- Celeb-DF-v2: official 518 test-video list, with source-video lineage exclusions in development; NOT identity-disjoint
- DFDC: author-defined source-group split of two publicly available training archives; NOT the challenge test

Detector architectures:
- Xception / DeepfakeBench `BACKBONE["xception"]` (real/fake two-class head)
- EfficientNet-B4 / DeepfakeBench `BACKBONE["efficientnetb4"]` (`efficientnet_pytorch`)
- ViT / `buildborderless/CommunityForensics-DeepfakeDet-ViT`, fine-tuned on FF++

The detector is trained/fine-tuned once on FF++ FIT, and then **frozen**. On Celeb/DFDC, the *same FF++ detector checkpoint* is used while TRIAGE FIT/CAL/VAL operate on labeled target partitions.

## 1. Environments and assets

Use a Python 3.10+ environment with the required PyTorch/CUDA build, then install `requirements-cnn.txt`, `code/requirements.txt`, and the dependencies of an **independent DeepfakeBench clone**. The ViT path additionally requires `transformers`, `huggingface_hub`, and `safetensors`; video preprocessing requires `opencv-python`, `dlib`, and DeepfakeBench's 81-landmark predictor.

DeepfakeBench source: https://github.com/SCLBD/DeepfakeBench

ImageNet **initialization**, not FF++ detector weights:

```bash
python -m triage_repro.fetch_imagenet --model xception --out-dir checkpoints
python -m triage_repro.fetch_imagenet --model efficientnetb4 --out-dir checkpoints
```

The downloader retrieves DeepfakeBench v1.0.0 `pretrained.zip`, extracts only the requested ImageNet backbone, and records SHA-256 hashes. Since the model weights and dataset images are third-party assets, **do not commit or redistribute them in this source repository**.

## 2. Dataset manifests and frames

The source scripts are:

- `code.preprocess_ffpp_dlib` — eight approximately uniformly spaced candidate frames/video, DeepfakeBench-compatible dlib alignment; frames with no detected face are skipped
- `code.prepare_ffpp_c23_manifest` — groups linked FF++ original IDs / source-target pairs, respects official train/val/test source splits, subdivides official train into TRIAGE FIT/CAL, and selects **two** retained face crops/video using `--frames-per-video 2`
- `code.prepare_external_video_manifest` — official Celeb test list with exclusion of same derived source-video groups; DFDC grouping by metadata's original real-video filename
- `code.preprocess_external_videos` — eight candidate frames, face-detection exclusions, up to two selected crops/video

Expected **frame** manifest columns: `path,label,split,group_id,video_id,method,official_split`. Labels 0=REAL, 1=FAKE; splits `fit,cal,val,test`. Public manifests must have paths converted to dataset-relative locations and undergo a licensing review. Do not publish private absolute paths.

Audit FF++ manifest:
```bash
python -m triage_repro.audit_manifest --manifest PATH_TO_FFPP_FRAME_MANIFEST.csv
```
Celeb/DFDC manifest:
```bash
python -m triage_repro.audit_external_manifest --manifest PATH_TO_EXTERNAL_MANIFEST.csv
```

## 3. Detector retraining

`triage_repro.make_detector_split` produces one frozen group-aware training/holdout split of **TRIAGE FF++ FIT**. With seed 42 and 15% of source groups held out, this should reproduce the ViT split if the exact same FIT manifest and algorithm are used; verify rather than assume the holdout is the same 860 frames.

For CNNs, `triage_repro.train_cnn` constructs the DeepfakeBench backbone, loads the correct ImageNet initialization, replaces the ImageNet classifier with a two-class head, fine-tunes the whole network, and saves the best detector-holdout AUROC checkpoint. The EfficientNet initialization restores the ImageNet input-stem weights overwritten by the standard upstream wrapper construction.

Proposed CNN settings: weighted balanced sampling, input 256x256 normalized to [-1, 1], horizontal flip p=0.5, Adam (learning rate 2e-5, decay 5e-4), no scheduler, cross-entropy, microbatch 4 × accumulation 8, maximum 10 epochs, seed 42. These are **new code defaults**, not historical CNN training facts. Every run writes a `training_report.json` and checkpoint hashes; the optimizer and augmentation must be reflected precisely in the eventual paper.

For ViT, `code.train_ffpp_vit_b7` loads the CommunityForensics ViT and fine-tunes using the FIT-only source-group protocol. It records a separate `training_report.json` and selects on detector holdout, not TRIAGE validation.

## 4. Unified experimental pipeline

The `configs/workflow.example.yaml` describes all nine detector/dataset combinations. Replace placeholders and run stages for each experiment from the repository root:

```bash
# Safe preview without GPU or datasets
python -m triage_repro.workflow --config configs/workflow.example.yaml --experiment xception_ffpp --stage select --dry-run
```

For each model, use its FF++ experiment for `audit`, `split`, and `train`. A single fine-tuned detector checkpoint is reused across all datasets. For each dataset, `select` evaluates candidate layers using **FIT/CAL/VAL only**. Following Appendix C, Xception and ViT use a common fixed-configuration layer screen (K=128, M=16, J=2, tau=1, weights 1/1/0.1) and then execute the full hyperparameter grid **only for the selected layer**. EfficientNet-B4 runs the same full validation grid for each candidate layer before selecting the layer. Selection is by validation AUROC, with AUPRC as the tie-breaker. `test` first refits TRIAGE and checks its selected **validation** AUROC and AUPRC; only then extracts and evaluates TEST.

The hyperparameter grid in the current code:
- `K ∈ {64,128,256,512}` prototypes, `M ∈ {4,8,16,32}` NMF topics per polarity
- `J ∈ {1,2,4}` reference profiles per predicted class
- `tau ∈ {0.01,0.1,1,10,100}`
- 11 score weight triples; seed 42, up to 50,000 prototype-fitting local features

The very large grid may require substantial local computation. `--stage select` writes and freezes the candidate metrics before a test run. Newly fine-tuned CNN checkpoints may lead to different selected layers and configurations from historical manuscript values; **never reuse the manuscript's original settings just to preserve earlier test numbers**.

## 5. TRIAGE fitting and final metrics

Authoritative scorer: `code.triage.TRIAGE`.

1. FIT: sample up to 50k local features, L2 normalize, MiniBatchKMeans into K prototypes; sum signed grad×activation into separate FAKE-supporting and REAL-supporting prototype histograms; fit separate coordinate-descent NMF topic bases.
2. CAL: encode topic distributions, learn predicted-class-specific profiles from correctly classified calibration examples, with class-specific k-means to produce up to J profiles.
3. VAL: select feature layer, K/M/J/tau and supporting/opposing/magnitude weights. The selected metric is error-ranking AUROC, not detector accuracy.
4. TEST: compute the risk score and additive per-topic contribution explanations, then compare with uncertainty (1−confidence), centroid distance, shrinkage Mahalanobis, kNN distance and prototype-attribution divergence.

`triage_repro.evaluate_final` writes:
- `final_test_metrics.json`: detector-error metrics, source-group bootstrap difference, 5/10/20% review queues with TP/FP/FN counts, low/high-confidence subgroup results, model selection and detector provenance
- `final_test_scores.csv`: per-frame confidence and six risk scores, with source/video IDs but no raw images
- `topic_explanations.jsonl`: reference index and signed topic contribution breakdown
- `fitted_triage.joblib`: fitted prototypes, NMF, and reference profiles (load **only** from trusted authors)

All are written to Git-ignored local `runs/` directories. The test evaluator refuses to overwrite previous final results and refuses to access TEST before validation replay.

## 6. Supporting analyses

Selected final-paper analyses migrated from the research repo:
- `code.evaluate_ffpp_reference_lomo`: held-out FF++ manipulation excluded from TRIAGE FIT/CAL/VAL; frozen detector, not an unseen-manipulation detector experiment
- `code.topic_sensitivity_v11`: attribution-prototype-bin deletion on sampled detector errors, score sensitivity only; random control is count-matched, **not** polarity-matched
- `code.report_practical_results`: review-budget and source-group bootstrap summaries
- `code.compare_seed_stability`, `code.seed_stability_v11`, `code.ablate_layers`, `code.compare_layers`: validation-only sensitivity and layer diagnostics
- `code.export_qualitative_cases` and `code.export_topic_prototype_gallery`: qualitative visual evidence, dependent on licensed local images

The historical `code.run_experiment` entry point predates the final nine-setting TIFS pipeline; the publication CLI is `triage_repro.workflow` plus `triage_repro.evaluate_final`.

## 7. Verification and release checklist

Automated CI runs dependency-light manifest/source-group tests, validation-layer provenance tests, Python syntax checks, and a dry-run of the unified CLI. Full image extraction, GPU training, and end-to-end detector evaluation require the authors' licensed data and GPUs; CI **does not** claim to reproduce the manuscript's numerical results.

Before making this repository public: rerun the CNNs from ImageNet initialization, verify the checkpoint weight provenance, regenerate 3×3 extraction/tuning/evaluation outputs, verify the paper's tables/figures against the new outputs, choose a software license, review DeepfakeBench/ViT/dataset redistribution terms, and publish only sanitized manifests and permitted score/model artifacts. In particular, avoid claiming identity-level disjointness for Celeb-DF-v2 or full-DFDC evaluation based on the two training archives.
