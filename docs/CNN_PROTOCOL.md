# CNN training protocol and reporting checklist

**Status: prospective configuration. No CNN weights have been trained by this repository yet.** Do not report these settings as the origin of earlier published detector scores.

## Architecture and initialization

- Xception: `SCLBD/DeepfakeBench` `BACKBONE["xception"]`, `mode="original"`, `inc=3`, `num_classes=2`, `dropout=False`. ImageNet initialization: `xception-b5690688.pth`; pointwise weights are adapted as required by the upstream detector initialization, with all non-head parameters checked.
- EfficientNet-B4: `SCLBD/DeepfakeBench` `BACKBONE["efficientnetb4"]`, backed by `efficientnet_pytorch.EfficientNet`, `num_classes=2`, `inc=3`, `dropout=False`. ImageNet initialization: `efficientnet-b4-6ed6700e.pth`.
- Both use the EXACT DeepfakeBench intermediate modules, not timm/torchvision substitutions. In the original paper, TRIAGE uses Xception `backbone.block8`, and B4 `backbone.efficientnet._blocks.29` (FF++) / `_blocks.15` (Celeb/DFDC).
- **Upstream EfficientNet-B4 nuance:** DeepfakeBench recreates `efficientnet._conv_stem` *after* loading ImageNet weights. Our trainer explicitly restores the matching ImageNet stem tensor, while preserving the DeepfakeBench architecture and its MBConv block indices. The **binary classification head** is newly initialized; the full model is then fine-tuned. This restoration is recorded in `training_report.json`.
- Both models produce two logits, one per class; class 0 REAL, class 1 FAKE.

## Advisor's six reproducibility questions

| Question | New Xception protocol | New EfficientNet-B4 protocol |
| --- | --- | --- |
| 1 Source | DeepfakeBench `xception` | DeepfakeBench `efficientnetb4` (`efficientnet_pytorch`) |
| 2 Data | FF++ c23 TRIAGE FIT only, with group-held-out subset reserved for detector selection; `WeightedRandomSampler` | Same |
| 3 Inputs | aligned 256x256 RGB face crops, channels scaled to [-1, 1], random horizontal flip p=0.5; no additional augmentation | Same |
| 4 Optimization | Full-network Adam, LR 2e-5, constant/no scheduler, weight decay 5e-4, microbatch 4, grad accumulation 8 (effective batch 32), 10 epochs, cross-entropy | Same |
| 5 Checkpoint | Best AUROC on frozen group-held-out detector images from FF++ FIT only | Same exact detector holdout |
| 6 Random seed | 42 | 42 |

**Important about dataset counts:** FF++ TRIAGE FIT has 5,752 frames in the manuscript. The saved ViT split protocol uses seed 42 and 15% of source groups as the detector holdout (860 holdout frames on that run). The new split script uses the same grouping algorithm; *verify* that the new `split_report.json` yields those exact images and group IDs. If you have the ViT's original `detector_train.csv` and `detector_holdout.csv`, compare file rows before claiming it is the *same* holdout. The training subsets are disjoint at source-group level from the detector holdout and from TRIAGE CAL/VAL/TEST, but not from TRIAGE FIT itself.

## Preparation

Clone [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench), install its dependencies, and get original **ImageNet** backbone initialization weights from the source documented in its `training/config/detector/*.yaml` files. Ensure neither weight file is a DeepfakeBench FF++-trained detector checkpoint.

To create the frozen split:

```powershell
python -m triage_repro.audit_manifest --manifest "D:\TRIAGE\ffpp_c23_manifest.csv"

python -m triage_repro.make_detector_split `
  --manifest "D:\TRIAGE\ffpp_c23_manifest.csv" `
  --seed 42 --holdout-fraction 0.15 `
  --out-dir "D:\TRIAGE\detector_split_seed42"
```

To train Xception:

```powershell
python -m triage_repro.train_cnn `
  --model xception --config configs/cnn_finetune.yaml `
  --source-manifest "D:\TRIAGE\ffpp_c23_manifest.csv" `
  --manifest-dir "D:\TRIAGE\detector_split_seed42" `
  --bench-dir "C:\path\to\DeepfakeBench" `
  --imagenet-weights "C:\path\to\DeepfakeBench\training\pretrained\xception-b5690688.pth" `
  --out-dir "runs\xception_seed42" --device cuda
```

To train EfficientNet-B4, change `--model efficientnetb4`, use `efficientnet-b4-6ed6700e.pth`, and choose `runs\efficientnetb4_seed42` as the output directory.

## What gets recorded

- `split_report.json`: source-manifest SHA, group counts, row counts, class counts, split seed.
- `training_report.json`: exact hyperparameters, class balancing, checkpoint provenance SHA, source/split/config hashes, validation metrics per epoch, best epoch.
- `<model>_best.pth`: complete two-class DeepfakeBench backbone weights and metadata; compatible with the strict TRIAGE adapters.

## Before updating the paper

1. Audit manifest counts, source groups, and original FF++ official split membership.
2. Ensure model initialization logs show the intended ImageNet weights and no unrelated missing backbone tensors.
3. Verify both detectors' final holdout performance and test performance.
4. Run new CNN attribution extraction using the trained checkpoints and the original TRIAGE source-group partitions.
5. Repeat validation-only layer/hyperparameter search and generate all frozen held-out tests (not just comparison tables).
6. Compare new manuscript statistics with newly produced JSON/CSV reports and update every dependent table, figure, and claim.
7. Never publish raw face frames, private original file paths, datasets, or checkpoints without license/provenance review.
