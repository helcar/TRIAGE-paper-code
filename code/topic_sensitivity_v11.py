"""B4 exploratory attribution-topic perturbation on frozen extracted arrays.

This tests *score sensitivity* to removal of topic-associated prototype
attributions. It does NOT establish causal faithfulness to detector logits.
Each topic is mapped to its highest-loading NMF prototypes; local attribution
at those prototypes is zeroed and the frozen TRIAGE model rescored.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .triage import TRIAGE, Config
from .run_experiment import read, pred
from .validate_triage import prototype_matrices
from .evaluation import error_labels


def calculate(model, positive, negative, prediction):
    encoded = model.encode(positive, negative)
    scores, explanations = model.score(encoded, prediction, explain=True)
    return scores, explanations


def run(a):
    fit, cal, test = [read(x) for x in (a.fit, a.cal, a.test)]
    cfg = Config(prototypes=a.prototypes, topics_positive=a.topics,
                 topics_negative=a.topics, references_real=a.references,
                 references_fake=a.references, tau=a.tau,
                 lambda_support=a.lambda_support, lambda_oppose=a.lambda_oppose,
                 lambda_magnitude=a.lambda_magnitude,
                 max_local_features=a.max_local_features, seed=a.seed)
    model = TRIAGE(cfg)
    batch = a.batch
    model.fit_prototypes(fit["features"][i:i+batch]
                         for i in range(0, len(fit["labels"]), batch))
    fp = prototype_matrices(model, fit, batch)
    cp = prototype_matrices(model, cal, batch)
    model.fit_topics(*fp)
    model.fit_references(model.encode(*cp), cal["labels"], pred(cal))
    tp, tn = prototype_matrices(model, test, batch)
    predictions = pred(test)
    y = error_labels(test["labels"], predictions)
    base, exp = calculate(model, tp, tn, predictions)
    if not np.isfinite(base).all():
        raise ValueError("Non-finite base scores")

    # Validate additive decomposition, including magnitude term.
    explained = np.array([
        sum(x["support_topic_contributions"]) +
        sum(x["oppose_topic_contributions"]) +
        x["magnitude_contribution"] for x in exp])
    deviation = float(np.max(np.abs(explained - base)))
    if deviation > 1e-7:
        raise RuntimeError(f"Score explanation decomposition mismatch: {deviation}")

    rng = np.random.default_rng(a.seed)
    indices = np.flatnonzero(y == 1)
    if len(indices) == 0:
        raise ValueError("No detector errors in TEST")
    if a.max_errors > 0 and len(indices) > a.max_errors:
        indices = np.sort(rng.choice(indices, size=a.max_errors, replace=False))

    original = []
    top_change = []
    random_change = []
    for i in indices:
        p = int(predictions[i])
        sup, opp = ("+", "-") if p == 1 else ("-", "+")
        x = exp[i]
        possibilities = [
            (sup, j, contribution) for j, contribution in
            enumerate(x["support_topic_contributions"])] + [
            (opp, j, contribution) for j, contribution in
            enumerate(x["oppose_topic_contributions"])]
        # Zero-weight topics have no contribution to the ranking, so exclude.
        possibilities = [v for v in possibilities if v[2] > 1e-12]
        if len(possibilities) < 2:
            continue
        strongest = max(possibilities, key=lambda v: v[2])
        random_topic = possibilities[int(rng.integers(len(possibilities)))]
        # Fair comparison: both interventions delete exactly N prototype bins.
        def intervene(choice):
            polarity, topic_id, _ = choice
            nmf_weights = model.nmf[polarity].components_[topic_id]
            columns = model.columns[polarity]
            bins = columns[np.argsort(nmf_weights)[-a.top_prototypes:]]
            pos = tp[i:i+1].copy()
            neg = tn[i:i+1].copy()
            (pos if polarity == "+" else neg)[0, bins] = 0.0
            result = model.score(model.encode(pos, neg), [p])
            return float(result[0])
        top_result = intervene(strongest)
        random_result = intervene(random_topic)
        original.append(float(base[i]))
        top_change.append(abs(top_result - base[i]))
        random_change.append(abs(random_result - base[i]))
    if not original:
        raise ValueError("No detector errors with at least two positive-weight topic contributions")
    top_change, random_change = np.asarray(top_change), np.asarray(random_change)
    return {
        "scope": "detector errors only; score sensitivity, not detector-logit causality",
        "n_test": int(len(y)), "n_errors": int(y.sum()),
        "errors_evaluated": int(len(original)),
        "max_additive_decomposition_error": deviation,
        "top_prototypes_per_topic": a.top_prototypes,
        "mean_absolute_risk_change_top_topic": float(top_change.mean()),
        "mean_absolute_risk_change_random_topic": float(random_change.mean()),
        "fraction_top_exceeds_random": float((top_change > random_change).mean()),
        "median_absolute_risk_change_top_topic": float(np.median(top_change)),
        "median_absolute_risk_change_random_topic": float(np.median(random_change)),
        "baseline_error_ranking_auroc": float(roc_auc_score(y, base)),
        "limitations": [
            "NMF topic/prototype assignments overlap and deletion affects total magnitude",
            "Attribution perturbation does not rerun the detector or measure changes in its logits",
            "Random topic is sampled among topics with nonzero weighted contributions",
            "Absolute risk-score change is a sensitivity measure, not a directional faithfulness measure",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("fit", "cal", "test", "output"):
        p.add_argument("--"+k, required=True)
    p.add_argument("--prototypes", type=int, default=64)
    p.add_argument("--topics", type=int, default=32)
    p.add_argument("--references", type=int, default=2)
    p.add_argument("--tau", type=float, default=.01)
    p.add_argument("--lambda-support", type=float, default=0.)
    p.add_argument("--lambda-oppose", type=float, default=1.)
    p.add_argument("--lambda-magnitude", type=float, default=.1)
    p.add_argument("--max-local-features", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--top-prototypes", type=int, default=5)
    p.add_argument("--max-errors", type=int, default=100)
    a = p.parse_args()
    if a.top_prototypes < 1 or a.top_prototypes > a.prototypes:
        p.error("--top-prototypes must be between 1 and --prototypes")
    result = run(a)
    output = Path(a.output.strip())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
