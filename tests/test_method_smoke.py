"""Scientific smoke test of the final TRIAGE attribution-topic score (CPU only).

Run with numpy, scipy, sklearn, torch and pytest installed.
Synthetic fixtures are not measurements of the paper's datasets.
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from code.triage import Config, TRIAGE
from code.evaluation import (FeatureBaselines, confidence_score, evaluate,
                             fit_prototype_profiles,
                             prototype_attribution_divergence)


def test_triage_fit_cal_val_test_synthetic():
    rng = np.random.default_rng(42)
    def features(n):
        x = rng.normal(size=(n, 9, 8)).astype(np.float32)
        a = rng.normal(size=(n, 9)).astype(np.float32)
        return x, a
    fit_f, fit_a = features(36)
    cal_f, cal_a = features(24)
    val_f, val_a = features(20)
    test_f, test_a = features(20)

    # Ensure CAL has correctly predicted samples of each class; VAL/TEST
    # contain both detector successes and mistakes.
    cal_labels = np.asarray([0] * 12 + [1] * 12)
    cal_preds = cal_labels.copy()
    test_labels = np.asarray([0] * 10 + [1] * 10)
    test_preds = test_labels.copy()
    test_preds[[2, 15]] = 1 - test_preds[[2, 15]]
    val_labels = test_labels.copy()
    val_preds = test_preds.copy()

    cfg = Config(prototypes=4, topics_positive=2, topics_negative=2,
                 references_real=1, references_fake=1,
                 max_local_features=256, seed=42)
    model = TRIAGE(cfg)
    model.fit_prototypes([fit_f])
    f_pos, f_neg = model.transform_prototypes(fit_f, fit_a)
    c_pos, c_neg = model.transform_prototypes(cal_f, cal_a)
    v_pos, v_neg = model.transform_prototypes(val_f, val_a)
    t_pos, t_neg = model.transform_prototypes(test_f, test_a)
    model.fit_topics(f_pos, f_neg)
    model.fit_references(model.encode(c_pos, c_neg), cal_labels, cal_preds)
    val_scores = model.score(model.encode(v_pos, v_neg), val_preds)
    scores, explanations = model.score(model.encode(t_pos, t_neg),
                                        test_preds, explain=True)
    assert np.all(np.isfinite(val_scores))
    assert np.all(np.isfinite(scores))
    assert len(explanations) == len(test_labels)
    for score, explanation in zip(scores, explanations):
        decomposed = (sum(explanation["support_topic_contributions"])
                      + sum(explanation["oppose_topic_contributions"])
                      + explanation["magnitude_contribution"])
        assert score == pytest.approx(decomposed, abs=1e-8)

    feature_baselines = FeatureBaselines().fit(
        cal_f.mean(axis=1), cal_labels, cal_preds)
    distances = feature_baselines.score(test_f.mean(axis=1), test_preds)
    pad = prototype_attribution_divergence(
        t_pos, t_neg, test_preds,
        fit_prototype_profiles(c_pos, c_neg, cal_labels, cal_preds))
    logits = (test_preds.astype(float) * 2 - 1) * 2
    risks = {
        "TRIAGE": scores,
        "Confidence": confidence_score(logits),
        "Centroid": distances["centroid"],
        "Mahalanobis": distances["mahalanobis"],
        "kNN": distances["knn"],
        "PAD": pad,
    }
    err = (test_preds != test_labels).astype(int)
    for risk in risks.values():
        output = evaluate(err, risk)
        assert output["n"] == len(test_labels)
        assert output["auroc"] is not None
        assert output["auprc"] is not None
