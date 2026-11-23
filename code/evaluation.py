"""Metrics and comparison baselines for error detection. Higher = more likely error."""
import numpy as np
from scipy.special import expit, rel_entr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from .triage import js_components


def confidence_score(logits):
    p = expit(np.asarray(logits))
    return np.minimum(p, 1 - p)


def entropy_score(logits):
    p = np.clip(expit(np.asarray(logits)), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1-p) * np.log1p(-p))


def error_labels(y, pred):
    return (np.asarray(y).astype(int) != np.asarray(pred).astype(int)).astype(int)


def risk_coverage(errors, score):
    e = np.asarray(errors)[np.argsort(score)]  # retain lowest-risk predictions
    n = len(e)
    coverage = np.arange(1, n+1) / n
    risk = np.cumsum(e) / np.arange(1, n+1)
    return coverage, risk


def evaluate(errors, score, budgets=(.05, .10, .20), coverages=(.5, .8, .9, 1.)):
    errors, score = np.asarray(errors, dtype=int), np.asarray(score, dtype=float)
    if errors.shape != score.shape or errors.ndim != 1 or len(errors) == 0:
        raise ValueError("errors and scores must be aligned nonempty 1D arrays")
    if not np.all(np.isfinite(score)):
        raise ValueError("Nonfinite scores")
    result = {"n": int(len(errors)), "error_prevalence": float(errors.mean())}
    if len(np.unique(errors)) == 2:
        result["auroc"] = float(roc_auc_score(errors, score))
        result["auprc"] = float(average_precision_score(errors, score))
        fpr, tpr, _ = roc_curve(errors, score)
        result["fpr_at_95_tpr"] = float(np.interp(.95, tpr, fpr))
    else:
        result.update(auroc=None, auprc=None, fpr_at_95_tpr=None)
    cov, risk = risk_coverage(errors, score)
    result["aurc"] = float(np.mean(risk))
    for coverage in coverages:
        n = max(1, int(np.ceil(coverage * len(errors))))
        result[f"risk_at_coverage_{coverage:g}"] = float(risk[n-1])
    for budget in budgets:
        k = max(1, int(np.ceil(budget * len(errors))))
        selected = np.argsort(score)[-k:]
        result[f"error_recall_at_budget_{budget:g}"] = (
            float(errors[selected].sum() / errors.sum()) if errors.sum() else None)
    return result


def prototype_attribution_divergence(positive, negative, predictions,
                                     profiles, eps=1e-3):
    """profiles: dict 0/1 -> {'+': K vector, '-': K vector}.
    Construct profiles ONLY from correctly predicted calibration data."""
    positive = np.asarray(positive, float)
    negative = np.asarray(negative, float)
    pp = (positive + eps) / (positive.sum(1, keepdims=True) + positive.shape[1] * eps)
    pn = (negative + eps) / (negative.sum(1, keepdims=True) + negative.shape[1] * eps)
    out = []
    for i, c in enumerate(predictions):
        sup, opp = ("+", "-") if c == 1 else ("-", "+")
        current = {"+": pp[i], "-": pn[i]}
        out.append(sum(js_components(current[q], profiles[int(c)][q]).sum() for q in (sup, opp)))
    return np.asarray(out)


def fit_prototype_profiles(positive, negative, labels, predictions, eps=1e-3):
    x = {}
    for q, arr in [("+", positive), ("-", negative)]:
        arr = np.asarray(arr, float)
        x[q] = (arr + eps) / (arr.sum(1, keepdims=True) + arr.shape[1] * eps)
    output = {}
    for c in (0, 1):
        mask = (np.asarray(labels) == c) & (np.asarray(predictions) == c)
        if not mask.any():
            raise ValueError(f"No correct samples for class {c}")
        output[c] = {q: x[q][mask].mean(0) for q in ("+", "-")}
    return output


class FeatureBaselines:
    """Fit ONLY on correctly predicted calibration features; use predicted class at test."""
    def fit(self, features, labels, predictions, k=5):
        f = np.asarray(features, float)
        y = np.asarray(labels)
        p = np.asarray(predictions)
        self.class_models = {}
        for c in (0, 1):
            sub = f[(y == c) & (p == c)]
            if len(sub) < 2:
                raise ValueError("Need at least two correct calibration samples per class")
            neighbors = NearestNeighbors(n_neighbors=min(k, len(sub))).fit(sub)
            cov = LedoitWolf().fit(sub)
            self.class_models[c] = (sub.mean(0), cov.precision_, neighbors)
        return self

    def score(self, features, predictions):
        f = np.asarray(features, float)
        scores = {"centroid": [], "mahalanobis": [], "knn": []}
        for vector, c in zip(f, predictions):
            mean, invcov, neighbors = self.class_models[int(c)]
            delta = vector - mean
            scores["centroid"].append(np.linalg.norm(delta))
            scores["mahalanobis"].append(np.sqrt(max(delta @ invcov @ delta, 0)))
            scores["knn"].append(float(neighbors.kneighbors(vector[None])[0][0].mean()))
        return {k: np.asarray(v) for k, v in scores.items()}


def fit_confidence_combination(val_triage_score, val_logits, val_errors):
    if len(np.unique(val_errors)) < 2:
        raise ValueError("Need both correct and erroneous validation predictions")
    features = np.column_stack((val_triage_score, confidence_score(val_logits)))
    return LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        features, val_errors)


def score_confidence_combination(model, triage_score, logits):
    features = np.column_stack((triage_score, confidence_score(logits)))
    return model.predict_proba(features)[:, 1]


def paired_bootstrap_auroc(errors, score_a, score_b, repeats=1000, seed=42):
    """Paired sample bootstrap of AUROC difference A - B."""
    rng = np.random.default_rng(seed)
    e = np.asarray(errors)
    a, b = np.asarray(score_a), np.asarray(score_b)
    differences = []
    for _ in range(repeats):
        idx = rng.integers(0, len(e), len(e))
        if len(np.unique(e[idx])) == 2:
            differences.append(roc_auc_score(e[idx], a[idx]) - roc_auc_score(e[idx], b[idx]))
    if not differences:
        return {"difference": None, "ci95": None}
    return {"difference": float(roc_auc_score(e, a) - roc_auc_score(e, b)),
            "ci95": np.quantile(differences, [.025, .975]).tolist()}
