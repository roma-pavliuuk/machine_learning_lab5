from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

OOD_DIR = Path(__file__).resolve().parents[2] / "models" / "ood"


def fit_segments(P_train: np.ndarray, c_range=range(2, 11), min_share: float = 0.05, random_state: int = 42) -> GaussianMixture:
    best, best_bic = None, np.inf
    for c in c_range:
        g = GaussianMixture(c, covariance_type="full", n_init=5, random_state=random_state, reg_covar=1e-4).fit(P_train)
        sizes = np.bincount(g.predict(P_train), minlength=c) / len(P_train)
        if sizes.min() < min_share:
            continue
        if g.bic(P_train) < best_bic:
            best, best_bic = g, g.bic(P_train)
    return best


def profile_segments(X_raw: pd.DataFrame, labels: np.ndarray, numeric_cols: List[str], categorical_cols: List[str],
                     top: int = 5) -> Dict[int, List[Dict]]:
    num = X_raw[numeric_cols].astype(float)
    cat = pd.get_dummies(X_raw[categorical_cols].astype("string").fillna("Missing"), prefix_sep="=").astype(float)
    med, iqr = num.median(), num.quantile(0.75) - num.quantile(0.25)
    out = {}
    for k in np.unique(labels):
        m = labels == k
        rows = []
        for col in numeric_cols:
            if iqr[col] > 0:
                rows.append({"feature": col, "kind": "z", "value": float((num.loc[m, col].median() - med[col]) / iqr[col]),
                             "segment": float(num.loc[m, col].median()), "global": float(med[col])})
        for col in cat.columns:
            rows.append({"feature": col, "kind": "share", "value": float(cat.loc[m, col].mean() - cat[col].mean()),
                         "segment": float(cat.loc[m, col].mean()), "global": float(cat[col].mean())})
        rows.sort(key=lambda r: -abs(r["value"]))
        out[int(k)] = rows[:top]
    return out


class Segmenter:
    def __init__(self, proj, gmm: GaussianMixture, names: Optional[Dict[str, str]] = None):
        self.proj, self.gmm, self.names = proj, gmm, names or {}

    def assign(self, S: np.ndarray) -> np.ndarray:
        return self.gmm.predict(self.proj.transform(S))

    def describe(self, segment_id: int) -> Dict:
        return {"id": int(segment_id), "name": self.names.get(str(int(segment_id)), f"сегмент {int(segment_id)}")}


def save_segmenter(gmm: GaussianMixture, names: Dict[str, str], ood_dir: Path = OOD_DIR) -> None:
    import skops.io as sio

    sio.dump(gmm, ood_dir / "segments_gmm.skops")
    (ood_dir / "segment_names.json").write_text(json.dumps(names, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_segmenter(proj, ood_dir: Path = OOD_DIR) -> Segmenter:
    import skops.io as sio

    path = ood_dir / "segments_gmm.skops"
    gmm = sio.load(str(path), trusted=sio.get_untrusted_types(file=str(path)))
    names = json.loads((ood_dir / "segment_names.json").read_text(encoding="utf-8")) if (ood_dir / "segment_names.json").exists() else {}
    return Segmenter(proj, gmm, names)
