"""One coherent paper workflow: train -> select on VAL -> evaluate on TEST.

The same script drives Xception, EfficientNet-B4, and ViT on all datasets,
without machine-specific paths in the repository. Heavy stages are run locally
by the authors; this CLI never invents or distributes the trained artifacts.

Configuration is in configs/workflow.example.yaml.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from .select_layer import choose


def run_command(argv, dry_run):
    print("+", " ".join(str(x) for x in argv), flush=True)
    if not dry_run:
        subprocess.run([str(x) for x in argv], check=True)


def slug(layer):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", layer).strip("_")


def load_experiment(path, name):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or name not in config.get("experiments", {}):
        raise ValueError(f"Unknown experiment {name}")
    exp = dict(config["experiments"][name])
    if exp["model"] not in ("xception", "efficientnetb4", "vit"):
        raise ValueError("Unsupported model")
    if exp["dataset"] not in ("FF++ c23", "Celeb-DF-v2", "DFDC"):
        raise ValueError("Unsupported dataset")
    if not exp.get("layers"):
        raise ValueError("At least one candidate layer is required")
    return config, exp


def extract_command(exp, layer, target, splits, authorize_test=False):
    model = exp["model"]
    if model == "xception":
        module = "code.extract_ffpp_large"
    elif model == "efficientnetb4":
        module = "code.extract_efficientnetb4"
    else:
        module = "code.extract_community_vit_b7"
    argv = [
        sys.executable, "-m", module, "--manifest", exp["manifest"],
        "--image-root", exp["image_root"], "--output-dir", str(target),
        "--layer", layer, "--device", exp.get("device", "cuda"),
        "--splits", *splits,
    ]
    if model in ("xception", "efficientnetb4"):
        argv += ["--checkpoint", exp["checkpoint"], "--bench-dir", exp["bench_dir"],
                 "--image-size", "256", "--batch-size", str(exp.get("extraction_batch", 2))]
    else:
        argv += ["--finetuned-checkpoint", exp["checkpoint"]]
        if exp.get("vit_revision"):
            argv += ["--revision", exp["vit_revision"]]
    if authorize_test:
        argv.append("--authorize-test")
    return argv


def tuning_command(exp, root, json_file, candidates_csv):
    return [
        sys.executable, "-m", "code.search_triage_sensitivity",
        "--fit", str(root / "fit"),
        "--cal", str(root / "cal"),
        "--val", str(root / "val"),
        "--output", str(json_file),
        "--candidates-csv", str(candidates_csv),
        "--k", "64", "128", "256", "512",
        "--m", "4", "8", "16", "32",
        "--j", "1", "2", "4",
        "--max-local-features", "50000",
        "--transform-batch", str(exp.get("triage_batch", 64)),
        "--seed", "42",
    ]


def evaluation_command(exp, root, grid, output_json, scores_csv, authorize_test=False):
    cmd = [
        sys.executable, "-m", "triage_repro.evaluate_final",
        "--manifest", exp["manifest"], "--arrays-root", str(root),
        "--selection-json", str(grid), "--model", exp["model"],
        "--dataset", exp["dataset"], "--output-json", str(output_json),
        "--scores-csv", str(scores_csv),
        "--transform-batch", str(exp.get("triage_batch", 64)),
    ]
    if exp["dataset"] == "FF++ c23":
        cmd.append("--ffpp-official")
    if authorize_test:
        cmd += [
            "--authorize-test",
            "--explanations-jsonl", str(Path(output_json).with_name("topic_explanations.jsonl")),
            "--fitted-model", str(Path(output_json).with_name("fitted_triage.joblib")),
        ]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--stage", choices=("audit", "split", "train", "select", "test", "all"),
                   required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved sequence without running GPU work")
    args = p.parse_args()
    global_cfg, exp = load_experiment(args.config, args.experiment)
    output_base = Path(exp.get("output_dir", "runs/" + args.experiment))
    selected_file = output_base / "selected_layer.json"

    def audit():
        if exp["dataset"] == "FF++ c23":
            cmd = [sys.executable, "-m", "triage_repro.audit_manifest",
                   "--manifest", exp["manifest"]]
            run_command(cmd, args.dry_run)
        else:
            # Check mandatory columns and group leakage without assuming FF++ official splits.
            cmd = [sys.executable, "-m", "triage_repro.audit_external_manifest",
                   "--manifest", exp["manifest"]]
            run_command(cmd, args.dry_run)

    def split():
        if exp["dataset"] != "FF++ c23":
            raise ValueError("Detector fine-tuning must use FF++ c23, not target datasets")
        run_command([
            sys.executable, "-m", "triage_repro.make_detector_split",
            "--manifest", exp["manifest"], "--out-dir", exp["detector_split_dir"],
            "--seed", "42", "--holdout-fraction", "0.15",
        ], args.dry_run)

    def train():
        if exp["dataset"] != "FF++ c23":
            raise ValueError("Train detectors once using the FF++ c23 experiment")
        if exp["model"] == "vit":
            cmd = [
                sys.executable, "-m", "code.train_ffpp_vit_b7",
                "--manifest", exp["manifest"],
                "--image-root", exp["image_root"],
                "--output-dir", exp["training_output_dir"],
                "--seed", "42", "--holdout-fraction", "0.15",
            ]
        else:
            cmd = [
                sys.executable, "-m", "triage_repro.train_cnn",
                "--model", exp["model"],
                "--config", global_cfg["cnn_training_config"],
                "--source-manifest", exp["manifest"],
                "--manifest-dir", exp["detector_split_dir"],
                "--image-root", exp["image_root"],
                "--bench-dir", exp["bench_dir"],
                "--imagenet-weights", exp["imagenet_weights"],
                "--out-dir", exp["training_output_dir"],
            ]
        run_command(cmd, args.dry_run)

    def select():
        candidates = []
        for layer in exp["layers"]:
            root = output_base / slug(layer)
            grid = root / "validation_grid.json"
            csv = root / "validation_candidates.csv"
            run_command(extract_command(exp, layer, root, ("fit", "cal", "val")), args.dry_run)
            candidates.append((layer, root, grid, csv))

        if exp["model"] in ("xception", "vit"):
            # Reproduce Appendix C: fixed-config layer screen BEFORE the
            # full TRIAGE hyperparameter grid, which is run once only.
            if args.dry_run:
                print("+ fixed-configuration VAL screen for:",
                      [(layer, str(root)) for layer, root, _, _ in candidates])
                chosen = candidates[0]
                print("+ [DRY RUN: subsequent full grid runs ONLY for winning layer]")
            else:
                if len(candidates) == 1:
                    chosen = candidates[0]
                    selected_file.parent.mkdir(parents=True, exist_ok=True)
                    if selected_file.exists():
                        raise FileExistsError(selected_file)
                    selected_file.write_text(json.dumps({
                        "selected_layer": chosen[0],
                        "selected_arrays_root": str(chosen[1]),
                        "selected_grid_json": str(chosen[2]),
                        "selection_rule": "Single prespecified layer",
                        "test_accessed": False,
                        "candidates": [],
                    }, indent=2), encoding="utf-8")
                else:
                    from .screen_layers_fixed import screen as screen_fixed_layers
                    selected = screen_fixed_layers(
                        [(layer, root) for layer, root, _, _ in candidates],
                        selected_file, batch=32)
                    chosen = next(entry for entry in candidates
                                  if entry[0] == selected["selected_layer"])
                    print("Fixed-screen selected layer:", selected["selected_layer"])
            run_command(tuning_command(exp, chosen[1], chosen[2], chosen[3]), args.dry_run)
        else:
            # v14 EfficientNet-B4 layer selection searches every layer with
            # the same validation grid and selects AUROC/AUPRC only.
            for _, root, grid, csv in candidates:
                run_command(tuning_command(exp, root, grid, csv), args.dry_run)
            if args.dry_run:
                print("+ full-grid VAL layer selection from:",
                      [(layer, str(root)) for layer, root, _, _ in candidates])
            elif len(candidates) == 1:
                selected_file.parent.mkdir(parents=True, exist_ok=True)
                if selected_file.exists():
                    raise FileExistsError(selected_file)
                selected_file.write_text(json.dumps({
                    "selected_layer": candidates[0][0],
                    "selected_arrays_root": str(candidates[0][1]),
                    "selected_grid_json": str(candidates[0][2]),
                    "selection_rule": "Single prespecified layer",
                    "test_accessed": False,
                    "candidates": [],
                }, indent=2), encoding="utf-8")
            else:
                result = choose([(layer, str(root), str(grid))
                                 for layer, root, grid, _ in candidates],
                                selected_file)
                print("Selected layer:", result["selected_layer"])

    def test():
        if args.dry_run:
            # A dry-run does not need the validation outputs to exist.
            layer = exp["layers"][0]
            root = output_base / slug(layer)
            grid = root / "validation_grid.json"
            print("+ [DRY RUN: actual layer comes from selected_layer.json]")
        else:
            if not selected_file.exists():
                raise FileNotFoundError(
                    f"No validation-only layer selection: {selected_file}. Run --stage select first.")
            picked = json.loads(selected_file.read_text(encoding="utf-8"))
            layer = picked["selected_layer"]
            root = Path(picked["selected_arrays_root"])
            grid = Path(picked["selected_grid_json"])
            if layer not in exp["layers"]:
                raise ValueError("Selected layer does not belong to this experiment")
            if not grid.is_file():
                raise FileNotFoundError(grid)
        final_json = output_base / "final_test_metrics.json"
        scores_csv = output_base / "final_test_scores.csv"
        # Refit and verify selected validation AUROC/AUPRC BEFORE touching TEST.
        run_command(evaluation_command(exp, root, grid, final_json, scores_csv), args.dry_run)
        run_command(extract_command(exp, layer, root, ("test",), authorize_test=True),
                    args.dry_run)
        run_command(evaluation_command(exp, root, grid, final_json, scores_csv,
                                       authorize_test=True), args.dry_run)

    if args.stage == "audit":
        audit()
    elif args.stage == "split":
        split()
    elif args.stage == "train":
        train()
    elif args.stage == "select":
        select()
    elif args.stage == "test":
        test()
    else:
        audit()
        if exp["dataset"] == "FF++ c23":
            split()
            train()
        select()
        test()


if __name__ == "__main__":
    main()
