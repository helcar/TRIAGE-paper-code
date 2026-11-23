"""TRIAGE: post-hoc attribution-topic misclassification scoring.

Input detector must return one fake-class logit per image. The selected layer
must output BCHW (CNN) or BTD (ViT) features. ViT CLS token can be excluded.
"""
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from scipy.special import rel_entr

EPS = 1e-3


def js_components(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = (p + q) / 2
    return 0.5 * (rel_entr(p, m) + rel_entr(q, m))


@dataclass
class Config:
    prototypes: int = 128
    topics_positive: int = 16
    topics_negative: int = 16
    references_real: int = 2
    references_fake: int = 2
    tau: float = 1.
    lambda_support: float = 1.
    lambda_oppose: float = 1.
    lambda_magnitude: float = 0.1
    eps: float = EPS
    seed: int = 42
    max_local_features: int = 200000
    exclude_cls_token: bool = False
    top_prototypes: int | None = None

    def validate(self):
        k = self.top_prototypes or self.prototypes
        if min(self.prototypes, self.topics_positive, self.topics_negative, k) < 1:
            raise ValueError("prototype and topic counts must be positive")
        if self.top_prototypes is not None and self.top_prototypes > self.prototypes:
            raise ValueError("top_prototypes cannot exceed prototypes")
        if max(self.topics_positive, self.topics_negative) > k:
            raise ValueError("NMF topic count must not exceed selected prototypes")
        if min(self.references_real, self.references_fake) < 1:
            raise ValueError("references must be positive")
        if self.tau <= 0 or self.eps <= 0:
            raise ValueError("tau and eps must be positive")
        if min(self.lambda_support, self.lambda_oppose, self.lambda_magnitude) < 0:
            raise ValueError("score weights must be nonnegative")


class DetectorAttributions:
    """A hook-backed extractor that does not update the detector weights."""
    def __init__(self, model, layer, device="cpu", exclude_cls_token=False):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.exclude_cls_token = exclude_cls_token
        self.activations = None
        self.hook = layer.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        self.activations = output
        if not output.requires_grad:
            raise ValueError("Layer activations must retain gradients")

    def close(self):
        self.hook.remove()

    def extract(self, images):
        # Input gradients keep the computational graph alive when detector
        # parameters are frozen; the model remains unchanged.
        x = images.to(self.device).detach().requires_grad_(True)
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits[:, 0]
            if logits.ndim != 1:
                raise ValueError("Detector must output a single fake-class logit per image")
            act = self.activations
            if act is None:
                raise RuntimeError("Hook did not capture activations")
            grad = torch.autograd.grad(logits.sum(), act)[0]
            if act.ndim == 4:
                features = act.permute(0, 2, 3, 1).reshape(act.shape[0], -1, act.shape[1])
                gradients = grad.permute(0, 2, 3, 1).reshape(grad.shape[0], -1, grad.shape[1])
            elif act.ndim == 3:
                features, gradients = act, grad
                if self.exclude_cls_token:
                    features, gradients = features[:, 1:], gradients[:, 1:]
            else:
                raise ValueError("Layer output must be BCHW or BTD")
            signed = (features * gradients).sum(-1)
            result = (logits.detach().cpu().numpy(),
                      features.detach().cpu().numpy(),
                      signed.detach().cpu().numpy())
        self.activations = None
        return result


class TRIAGE:
    def __init__(self, config=None):
        self.cfg = config or Config()
        self.cfg.validate()
        self.prototypes = None
        self.nmf = {}
        self.columns = {}
        self.profiles = {}
        self.scalers = {}

    def fit_prototypes(self, feature_batches):
        """feature_batches: iterable of [batch, positions, channels] numpy arrays."""
        rng = np.random.default_rng(self.cfg.seed)
        limit = self.cfg.max_local_features
        reservoir = []
        count = 0
        # Reservoir sampling keeps bounded memory across all feature batches.
        for batch in feature_batches:
            z = np.array(batch, dtype=np.float32, copy=True).reshape(-1, batch.shape[-1])
            z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
            for row in z:
                count += 1
                if len(reservoir) < limit:
                    reservoir.append(row.copy())
                else:
                    j = int(rng.integers(count))
                    if j < limit:
                        reservoir[j] = row.copy()
        if len(reservoir) < self.cfg.prototypes:
            raise ValueError("Not enough local features to fit prototypes")
        self.prototypes = MiniBatchKMeans(
            n_clusters=self.cfg.prototypes, random_state=self.cfg.seed,
            n_init=3, batch_size=4096).fit(np.asarray(reservoir))

    def transform_prototypes(self, features, attribution):
        """Return [N,K] fake-supporting and real-supporting attribution."""
        if self.prototypes is None:
            raise RuntimeError("Fit prototypes first")
        x = np.asarray(features)
        a = np.asarray(attribution)
        if x.shape[:2] != a.shape:
            raise ValueError("Attribution shape must match local features")
        flat = x.reshape(-1, x.shape[-1]).copy()
        flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
        assign = self.prototypes.predict(flat).reshape(a.shape)
        plus = np.zeros((a.shape[0], self.cfg.prototypes))
        minus = np.zeros_like(plus)
        for i in range(a.shape[0]):
            np.add.at(plus[i], assign[i], np.maximum(a[i], 0))
            np.add.at(minus[i], assign[i], np.maximum(-a[i], 0))
        return plus, minus

    def fit_topics(self, positive, negative):
        for key, data, m in [("+", positive, self.cfg.topics_positive),
                              ("-", negative, self.cfg.topics_negative)]:
            x = np.asarray(data, dtype=float)
            if x.ndim != 2 or x.shape[1] != self.cfg.prototypes:
                raise ValueError("Expected [N,K] prototype-attribution matrix")
            if self.cfg.top_prototypes is None:
                columns = np.arange(x.shape[1])
            else:
                presence = (x > 0).mean(0)
                conditional = x.sum(0) / np.maximum((x > 0).sum(0), 1)
                ranking = conditional * np.maximum(presence, .05)
                columns = np.argsort(ranking)[-self.cfg.top_prototypes:]
            self.columns[key] = columns
            self.nmf[key] = NMF(
                n_components=m, init="nndsvda", solver="cd",
                max_iter=500, tol=1e-4, random_state=self.cfg.seed
            ).fit(x[:, columns] + 1e-10)

    def encode(self, positive, negative):
        result = {}
        for key, raw in [("+", positive), ("-", negative)]:
            if key not in self.nmf:
                raise RuntimeError("Fit NMF topics before encoding")
            x = np.asarray(raw, dtype=float)
            magnitude = x.sum(axis=1)
            w = self.nmf[key].transform(x[:, self.columns[key]] + 1e-10)
            p = (w + self.cfg.eps) / (w.sum(1, keepdims=True) + w.shape[1] * self.cfg.eps)
            result[key] = {"p": p, "a": magnitude,
                           "eta": magnitude / (magnitude + self.cfg.tau)}
        return result

    @staticmethod
    def joint(encoded):
        return np.column_stack([encoded["+"]["p"], encoded["-"]["p"],
            np.log1p(encoded["+"]["a"]), np.log1p(encoded["-"]["a"])])

    def fit_references(self, encoded, labels, predictions):
        y, pred = np.asarray(labels), np.asarray(predictions)
        if not np.array_equal(np.unique(y), np.intersect1d(np.unique(y), [0, 1])):
            raise ValueError("Binary labels required")
        v = self.joint(encoded)
        for c, n_ref in [(0, self.cfg.references_real), (1, self.cfg.references_fake)]:
            idx = np.flatnonzero((y == c) & (pred == c))
            if len(idx) < n_ref:
                raise ValueError(f"Not enough correct calibration samples of class {c}")
            scaler = StandardScaler().fit(v[idx])
            self.scalers[c] = scaler
            groups = KMeans(n_clusters=n_ref, n_init=10, random_state=self.cfg.seed).fit_predict(
                scaler.transform(v[idx]))
            profiles = []
            for j in range(n_ref):
                ids = idx[groups == j]
                profiles.append({
                    "+": encoded["+"]["p"][ids].mean(0),
                    "-": encoded["-"]["p"][ids].mean(0),
                    "mu+": np.log1p(encoded["+"]["a"][ids]).mean(),
                    "mu-": np.log1p(encoded["-"]["a"][ids]).mean(),
                    "n": len(ids),
                })
            self.profiles[c] = profiles

    def score(self, encoded, predictions, explain=False):
        predictions = np.asarray(predictions, dtype=int)
        output = np.empty(len(predictions))
        explanations = []
        for i, c in enumerate(predictions):
            if c not in self.profiles:
                raise ValueError(f"No reference profiles for class {c}")
            sup, opp = ("+", "-") if c == 1 else ("-", "+")
            candidates = []
            for j, ref in enumerate(self.profiles[c]):
                s = (encoded[sup]["eta"][i] *
                     js_components(encoded[sup]["p"][i], ref[sup]))
                o = (encoded[opp]["eta"][i] *
                     js_components(encoded[opp]["p"][i], ref[opp]))
                magnitude = sum(abs(np.log1p(encoded[q]["a"][i]) - ref["mu" + q])
                                for q in ("+", "-"))
                score = (self.cfg.lambda_support * s.sum()
                         + self.cfg.lambda_oppose * o.sum()
                         + self.cfg.lambda_magnitude * magnitude)
                candidates.append((score, j, s, o, magnitude))
            best = min(candidates, key=lambda z: z[0])
            output[i] = best[0]
            if explain:
                ref = self.profiles[c][best[1]]
                explanations.append({
                    "reference_index": best[1],
                    "support_topic_contributions": (self.cfg.lambda_support * best[2]).tolist(),
                    "oppose_topic_contributions": (self.cfg.lambda_oppose * best[3]).tolist(),
                    "support_topic_delta": (encoded[sup]["p"][i] - ref[sup]).tolist(),
                    "oppose_topic_delta": (encoded[opp]["p"][i] - ref[opp]).tolist(),
                    "magnitude_contribution": float(self.cfg.lambda_magnitude * best[4]),
                })
        return (output, explanations) if explain else output
