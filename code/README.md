# TRIAGE scientific implementation modules

**Authoritative publication workflow:** `triage_repro.workflow` (training, feature-layer screening, validation tuning, and held-out testing) and `triage_repro.evaluate_final` (final misclassification-ranking metrics).

This directory contains code migrated from the original TRIAGE research repository and selected for the final TIFS study.

## Scientific method

- `triage.py`: signed gradient × activation evidence, detector-local prototype clustering, polarity-specific NMF topics, multi-reference calibration, topic-decomposable risk scoring.
- `xception_adapter.py` and `efficientnetb4_adapter.py`: DeepfakeBench binary detector backbones with strict checkpoint loading.
- `extract_ffpp_large.py`, `extract_efficientnetb4.py`, `extract_community_vit_b7.py`: extraction of frozen model logits, activations, and signed gradient attributions.
- `evaluation.py`: confidence, centroid, Mahalanobis, kNN and prototype-attribution divergence baselines.
- `search_triage_sensitivity.py`: FIT/CAL/VAL-only model selection.
- `train_ffpp_vit_b7.py`: FF++ FIT-only fine-tuning of the CommunityForensics ViT.

## Dataset and supplemental analyses

- Source-group-aware video and frame manifests: `prepare_ffpp_c23_manifest.py`, `preprocess_ffpp_dlib.py`, `prepare_external_video_manifest.py`, `preprocess_external_videos.py`.
- Final-paper supplementary analyses: `evaluate_ffpp_reference_lomo.py`, `topic_sensitivity_v11.py`, `report_practical_results.py`, `group_bootstrap.py`, `seed_stability_v11.py`, and layer-selection, qualitative and performance diagnostics.

Historical utility `run_experiment.py` is retained as an import provider, but it is **not** the final nine-setting benchmark entry point. Use the staged publication pipeline instead.

**Important:** Newly ImageNet-initialized CNN checkpoints must be trained and all related TRIAGE scores regenerated before the manuscript's numerical results can be updated. See [reproduction and release checklist](../docs/REPRODUCIBILITY.md).
