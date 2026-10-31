from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

_GROUP_PREFIX = re.compile(r"^[a-z_]+__")


def aggregate_name(feature: str, request_fields: List[str], categorical: List[str]) -> str:
    name = _GROUP_PREFIX.sub("", feature)
    if name.endswith("_missing") and name[:-8] in request_fields:
        return name[:-8]
    for col in sorted(categorical, key=len, reverse=True):
        if name == col or name.startswith(col + "_"):
            return col
    return name


class LocalExplainer:
    def __init__(self, export_pipe, background: np.ndarray, feature_names: List[str], request_fields: List[str],
                 categorical: List[str], positive_index: int = 1):
        import shap
        from lime.lime_tabular import LimeTabularExplainer

        self.pipe = export_pipe
        self.model = export_pipe[-1]
        self.head = export_pipe[:-1]
        self.feature_names = list(feature_names)
        self.request_fields, self.categorical = request_fields, categorical
        self.positive = positive_index
        self.background = background.astype(np.float32)
        bg_model = self._to_model_space(self.background)
        self.model_feature_names = self._selected_names()
        self.shap = shap.TreeExplainer(self.model, data=shap.maskers.Independent(bg_model, max_samples=len(bg_model)),
                                       feature_perturbation="interventional", model_output="probability")
        self.lime = LimeTabularExplainer(training_data=self.background, feature_names=self.feature_names,
                                         class_names=["good", "bad"], mode="classification",
                                         discretize_continuous=False, random_state=42)

    def _to_model_space(self, Z: np.ndarray) -> np.ndarray:
        P = self.head.transform(Z) if len(self.pipe.steps) > 1 else Z
        return np.asarray(P.toarray() if hasattr(P, "toarray") else P, dtype=np.float64)

    def _selected_names(self) -> List[str]:
        names = np.asarray(self.feature_names)
        for _, step in self.head.steps:
            if hasattr(step, "transformers_"):
                idx = [int(i) for _, trans, cols in step.transformers_ if trans != "drop" for i in list(cols)]
                names = names[idx]
            elif hasattr(step, "get_support"):
                names = names[step.get_support()]
        return [str(n) for n in names]

    def _aggregate(self, names: List[str], values: np.ndarray, data: np.ndarray = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
        agg: Dict[str, float] = {}
        raw: Dict[str, float] = {}
        for i, (n, v) in enumerate(zip(names, values)):
            key = aggregate_name(n, self.request_fields, self.categorical)
            agg[key] = agg.get(key, 0.0) + float(v)
            if data is not None and key not in raw:
                raw[key] = float(data[i])
        keys = list(agg)
        return keys, np.array([agg[k] for k in keys]), np.array([raw.get(k, np.nan) for k in keys])

    def explain(self, z: np.ndarray, record: Dict, lime_samples: int = 1000) -> Dict:
        z = np.asarray(z, dtype=np.float32).reshape(1, -1)
        x_model = self._to_model_space(z)
        sv = self.shap.shap_values(x_model)
        sv = np.asarray(sv)
        values = sv[0, :, self.positive] if sv.ndim == 3 else sv[0]
        base = self.shap.expected_value
        base = float(base[self.positive]) if np.ndim(base) else float(base)
        names, agg_values, _ = self._aggregate(self.model_feature_names, values, x_model[0])
        data = [record.get(k, None) if k in record else None for k in names]

        lime_exp = self.lime.explain_instance(z[0], lambda Z: self.pipe.predict_proba(np.asarray(Z, dtype=np.float32)),
                                              num_features=len(self.feature_names), num_samples=lime_samples, labels=(self.positive,))
        lime_pairs = lime_exp.as_map()[self.positive]
        lime_names, lime_values, _ = self._aggregate([self.feature_names[i] for i, _ in lime_pairs],
                                                     np.array([w for _, w in lime_pairs]))
        lime_sorted = sorted(zip(lime_names, lime_values), key=lambda t: -abs(t[1]))

        shap_rank = {k: abs(v) for k, v in zip(names, agg_values)}
        lime_rank = {k: abs(v) for k, v in lime_sorted}
        common = sorted(set(shap_rank) | set(lime_rank))
        top_shap = {k for k, _ in sorted(shap_rank.items(), key=lambda t: -t[1])[:5]}
        top_lime = {k for k, _ in lime_sorted[:5]}
        rho = spearmanr([shap_rank.get(k, 0.0) for k in common], [lime_rank.get(k, 0.0) for k in common]).statistic
        return {
            "shap": {"values": [float(v) for v in agg_values], "base_value": base, "data": data, "feature_names": names,
                     "prediction": float(base + agg_values.sum())},
            "lime": [(k, float(v)) for k, v in lime_sorted],
            "agreement": {"top5_overlap": len(top_shap & top_lime) / 5, "spearman": float(rho) if rho == rho else None},
        }
