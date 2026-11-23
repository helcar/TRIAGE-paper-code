# TRIAGE

Code for **TRIAGE: Explainable Error Ranking for Face Deepfake Detectors**.

This repository implements the face-deepfake detector training, feature-attribution extraction, TRIAGE error-risk scoring, and evaluation used by the paper. Dataset files and pretrained weights are obtained separately.

## Main scripts

| File | Purpose |
| --- | --- |
| `triage_repro/workflow.py` | Main entry point: dataset audit, detector training, layer selection, TRIAGE tuning and evaluation |
| `triage_repro/train_cnn.py` | Fine-tune DeepfakeBench Xception and EfficientNet-B4 from ImageNet-pretrained weights |
| `triage_repro/fetch_imagenet.py` | Download the original ImageNet backbone initialization weights |
| `code/train_ffpp_vit_b7.py` | Fine-tune the CommunityForensics ViT on FF++ |
| `code/preprocess_ffpp_dlib.py` | Extract and align FF++ face frames |
| `code/prepare_ffpp_c23_manifest.py` | Build FF++ source-group-aware FIT/CAL/VAL/TEST manifest |
| `code/preprocess_external_videos.py` | Extract and align Celeb-DF-v2 and DFDC frames |
| `code/prepare_external_video_manifest.py` | Build Celeb-DF-v2 / DFDC manifests |
| `triage_repro/make_detector_split.py` | Make source-group-separated CNN training/holdout sets |
| `code/extract_ffpp_large.py` | Extract Xception logits, intermediate features and signed attributions |
| `code/extract_efficientnetb4.py` | Extract EfficientNet-B4 logits, features and signed attributions |
| `code/extract_community_vit_b7.py` | Extract ViT logits, features and signed attributions |
| `code/triage.py` | TRIAGE algorithm: prototypes, polarity-specific NMF topics, reference profiles and risk score |
| `code/search_triage_sensitivity.py` | Choose TRIAGE hyperparameters using validation data |
| `triage_repro/screen_layers_fixed.py` | Fixed-configuration layer selection for Xception and ViT |
| `triage_repro/select_layer.py` | Validation-based EfficientNet-B4 layer selection |
| `triage_repro/evaluate_final.py` | Final six-method misclassification ranking, explanations, queues and bootstrap metrics |
| `code/evaluate_ffpp_reference_lomo.py` | FF++ leave-one-manipulation-out TRIAGE analysis |
| `code/topic_sensitivity.py` | Topic-attribution perturbation analysis |

The remaining Python files are small helpers imported by these modules: detector adapters, frame loaders, manifest validation, prototype transformations and baseline metrics.

## Setup

Install packages from `requirements.txt`, a compatible PyTorch build, and the separately maintained [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) package and dependencies. ViT uses Hugging Face Transformers. Supply your own licensed FF++, Celeb-DF-v2 and DFDC datasets.

Set the paths in `configs/workflow.example.yaml`. `configs/cnn_finetune.yaml` specifies the CNN fine-tuning settings.

## Run

```bash
python -m triage_repro.workflow --config configs/workflow.example.yaml --experiment xception_ffpp --stage all
```

Experiment names follow `xception_ffpp`, `effb4_celeb`, `vit_dfdc`, etc. Available stages: `audit`, `split`, `train`, `select`, `test`, `all`. Train each detector on FF++ once, then reuse its frozen checkpoint on Celeb-DF-v2 and DFDC. The TEST partition is evaluated only after validation selection.

The source code does not include dataset images, ImageNet weights, trained checkpoints or precomputed experimental scores.
