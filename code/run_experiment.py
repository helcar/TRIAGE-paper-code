"""Run TRIAGE experiments on precomputed detector outputs.

NPZ files must contain aligned arrays:
 logits [N], labels [N], features [N,T,D], attributions [N,T].
 Detector outputs and gradients are generated with DetectorAttributions.
 Separate NPZ files MUST correspond to disjoint source videos/identities.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .triage import TRIAGE, Config
from .evaluation import (evaluate, error_labels, confidence_score,
    FeatureBaselines, fit_prototype_profiles, prototype_attribution_divergence,
    fit_confidence_combination, score_confidence_combination,
    paired_bootstrap_auroc)


def read(path):
    path = Path(path)
    if path.is_dir():
        if not (path / "_complete.json").is_file():
            raise ValueError(f"Incomplete extraction: {path}")
        data = {k: np.load(path / f"{k}.npy", mmap_mode="r", allow_pickle=False)
                for k in ("logits", "labels", "features", "attributions")}
    else:
        with np.load(path, allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
    for name in ("logits", "labels", "features", "attributions"):
        if name not in data:
            raise ValueError(f"{path}: missing {name}")
    n = len(data["logits"])
    if (len(data["labels"]) != n or len(data["features"]) != n or
            len(data["attributions"]) != n):
        raise ValueError(f"{path}: unequal sample counts")
    return data


def pred(d):
    return (np.asarray(d["logits"]) >= 0).astype(int)


def pooled_features(d):
    return d["features"].mean(axis=1)


def run(splits, cfg):
    """Returns experiment report. Test split is never used for model fitting."""
    model = TRIAGE(cfg)
    fit, cal, val, test = [read(splits[k]) for k in ("fit", "cal", "val", "test")]
    model.fit_prototypes([fit["features"]])
    proto = {k: model.transform_prototypes(d["features"], d["attributions"])
             for k, d in [("fit", fit), ("cal", cal), ("val", val), ("test", test)]}
    model.fit_topics(*proto["fit"])
    enc = {k: model.encode(*proto[k]) for k in proto}
    model.fit_references(enc["cal"], cal["labels"], pred(cal))

    val_error = error_labels(val["labels"], pred(val))
    test_error = error_labels(test["labels"], pred(test))
    val_score = model.score(enc["val"], pred(val))
    scores = {"TRIAGE": model.score(enc["test"], pred(test)),
              "Confidence": confidence_score(test["logits"])}
    feat = FeatureBaselines().fit(pooled_features(cal), cal["labels"], pred(cal))
    scores.update({"Feature_" + k: v for k, v in
                   feat.score(pooled_features(test), pred(test)).items()})
    reference = fit_prototype_profiles(*proto["cal"], cal["labels"], pred(cal))
    scores["Prototype_Attribution_Divergence"] = prototype_attribution_divergence(
        *proto["test"], pred(test), reference)
    if len(np.unique(val_error)) == 2:
        combiner = fit_confidence_combination(val_score, val["logits"], val_error)
        scores["TRIAGE_plus_Confidence"] = score_confidence_combination(
            combiner, scores["TRIAGE"], test["logits"])
    metrics = {name: evaluate(test_error, value) for name, value in scores.items()}
    analyses = {}
    if len(np.unique(test_error)) == 2:
        analyses["triage_vs_confidence"] = paired_bootstrap_auroc(
            test_error, scores["TRIAGE"], scores["Confidence"])
    for c, name in [(0, "predicted_real"), (1, "predicted_fake")]:
        idx = pred(test) == c
        if idx.any():
            analyses[name] = {key: evaluate(test_error[idx], v[idx])
                              for key, v in scores.items()}
    high_conf = (1 - confidence_score(test["logits"])) >= .9
    if high_conf.any():
        analyses["high_confidence"] = {
            "count": int(high_conf.sum()),
            "errors": int(test_error[high_conf].sum()),
            "triage": evaluate(test_error[high_conf], scores["TRIAGE"][high_conf])
        }
    return {"config": asdict(cfg), "test_metrics": metrics, "analyses": analyses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("fit", "cal", "val", "test"):
        parser.add_argument("--" + split, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prototypes", type=int, default=128)
    parser.add_argument("--topics", type=int, default=16)
    parser.add_argument("--references", type=int, default=2)
    parser.add_argument("--tau", type=float, default=1.)
    parser.add_argument("--lambda-support", type=float, default=1.)
    parser.add_argument("--lambda-oppose", type=float, default=1.)
    parser.add_argument("--lambda-magnitude", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = Config(prototypes=args.prototypes, topics_positive=args.topics,
                 topics_negative=args.topics, references_real=args.references,
                 references_fake=args.references, tau=args.tau,
                 lambda_support=args.lambda_support,
                 lambda_oppose=args.lambda_oppose,
                 lambda_magnitude=args.lambda_magnitude, seed=args.seed)
    report = run({key: getattr(args, key) for key in ("fit", "cal", "val", "test")}, cfg)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
