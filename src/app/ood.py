from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OOD_DIR = Path(__file__).resolve().parents[2] / "models" / "ood"
RULE = {"checking": "<0 DM", "duration_gt": 24, "savings": "<100 DM"}


def non_ml_rule(record: Dict) -> int:
    checking = record.get("checking_account")
    savings = record.get("savings_account")
    duration = record.get("duration") or 0
    return int(checking == RULE["checking"] or (duration > RULE["duration_gt"] and savings == RULE["savings"]))


def model_space(export_pipe, Z: np.ndarray) -> np.ndarray:
    P = export_pipe[:-1].transform(Z) if len(export_pipe.steps) > 1 else Z
    return P.toarray() if hasattr(P, "toarray") else np.asarray(P, dtype=float)


def fit_projection(S_fit: np.ndarray, max_components: int = 10, random_state: int = 42) -> Pipeline:
    evr = np.cumsum(PCA(random_state=random_state).fit(StandardScaler().fit_transform(S_fit)).explained_variance_ratio_)
    k = min(int(np.searchsorted(evr, 0.9)) + 1, max_components, S_fit.shape[1])
    proj = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=k, whiten=True, random_state=random_state))])
    return proj.fit(S_fit)


def fit_kde(P_fit: np.ndarray, random_state: int = 42):
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    lo, hi = -1.0, 1.0
    for _ in range(4):
        grid = np.logspace(lo, hi, 20)
        search = GridSearchCV(KernelDensity(), {"bandwidth": grid}, cv=cv).fit(P_fit)
        h = search.best_params_["bandwidth"]
        if np.isclose(h, grid[0]):
            lo -= 1
        elif np.isclose(h, grid[-1]):
            hi += 1
        else:
            return search.best_estimator_, {"bandwidth": float(h), "grid": [float(grid[0]), float(grid[-1])], "cv_loglik": float(search.best_score_)}
    return search.best_estimator_, {"bandwidth": float(h), "grid": [float(grid[0]), float(grid[-1])], "cv_loglik": float(search.best_score_), "edge": True}


def fit_gmm(P_fit: np.ndarray, c_range=range(1, 11), random_state: int = 42):
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = {}
    for c in c_range:
        ll = []
        for tr, va in cv.split(P_fit):
            g = GaussianMixture(c, covariance_type="full", n_init=3, random_state=random_state, reg_covar=1e-4).fit(P_fit[tr])
            ll.append(g.score(P_fit[va]))
        scores[c] = float(np.mean(ll))
    best_c = max(scores, key=scores.get)
    gmm = GaussianMixture(best_c, covariance_type="full", n_init=10, random_state=random_state, reg_covar=1e-4).fit(P_fit)
    return gmm, {"n_components": int(best_c), "cv_loglik": scores, "edge": best_c == max(c_range)}


@dataclass
class OODDetector:
    proj: Pipeline
    density: object
    threshold: float
    kind: str

    def log_density(self, S: np.ndarray) -> np.ndarray:
        return self.density.score_samples(self.proj.transform(S))

    def is_ood(self, S: np.ndarray) -> np.ndarray:
        return self.log_density(S) < self.threshold


def synthetic_ood(proj: Pipeline, S_test: np.ndarray, sigma: float = 3.0, axes=(0, 1), random_state: int = 42) -> np.ndarray:
    P = proj.transform(S_test)
    P_shift = P.copy()
    rng = np.random.default_rng(random_state)
    signs = rng.choice([-1, 1], size=len(P))
    for ax in axes:
        P_shift[:, ax] += signs * sigma * P[:, ax].std()
    pca, scale = proj.named_steps["pca"], proj.named_steps["scale"]
    return scale.inverse_transform(pca.inverse_transform(P_shift))


def build_detectors(S_train: np.ndarray, S_test: np.ndarray, random_state: int = 42) -> Dict:
    S_fit, S_cal = train_test_split(S_train, test_size=0.2, random_state=random_state)
    proj = fit_projection(S_fit, random_state=random_state)
    P_fit = proj.transform(S_fit)
    evr = float(proj.named_steps["pca"].explained_variance_ratio_.sum())
    detectors, report = {}, {"projection": {"n_components": int(proj.named_steps["pca"].n_components_), "explained_variance": evr,
                                            "model_space_dim": int(S_train.shape[1])}}
    kde, kde_info = fit_kde(P_fit, random_state)
    gmm, gmm_info = fit_gmm(P_fit, random_state=random_state)
    S_synth = synthetic_ood(proj, S_test, random_state=random_state)
    for kind, dens, info in (("kde", kde, kde_info), ("gmm", gmm, gmm_info)):
        thr = float(np.percentile(dens.score_samples(proj.transform(S_cal)), 1))
        det = OODDetector(proj, dens, thr, kind)
        detectors[kind] = det
        report[kind] = {**info, "threshold": thr,
                        "fallback_share_test": float(det.is_ood(S_test).mean()),
                        "synthetic_ood_detected_share": float(det.is_ood(S_synth).mean())}
    return {"detectors": detectors, "report": report, "S_cal": S_cal}


def save_detectors(result: Dict, selected: str, extra: Optional[Dict] = None) -> None:
    import skops.io as sio

    OOD_DIR.mkdir(parents=True, exist_ok=True)
    for kind, det in result["detectors"].items():
        sio.dump(det.proj, OOD_DIR / f"{kind}_projection.skops")
        sio.dump(det.density, OOD_DIR / f"{kind}_density.skops")
    meta = {"selected": selected, "report": result["report"], "rule": RULE, **(extra or {})}
    (OOD_DIR / "ood_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_detector(ood_dir: Path = OOD_DIR) -> OODDetector:
    import skops.io as sio

    meta = json.loads((ood_dir / "ood_meta.json").read_text(encoding="utf-8"))
    kind = meta["selected"]
    load = lambda name: sio.load(str(ood_dir / name), trusted=sio.get_untrusted_types(file=str(ood_dir / name)))
    return OODDetector(load(f"{kind}_projection.skops"), load(f"{kind}_density.skops"),
                       float(meta["report"][kind]["threshold"]), kind)
