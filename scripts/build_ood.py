import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
import yaml
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.app.ood import (OOD_DIR, build_detectors, model_space, non_ml_rule, save_detectors)
from src.app.segments import fit_segments, profile_segments, save_segmenter
from src.features.lab3_bridge import total_cost


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config/config_rf.yaml").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "models/model_meta.json").read_text(encoding="utf-8"))
    pipe_path = ROOT / "models/pipeline.skops"
    export_pipe = sio.load(str(pipe_path), trusted=sio.get_untrusted_types(file=str(pipe_path)))
    Z_train, Z_test = np.load(ROOT / "data/train_features.npy"), np.load(ROOT / "data/test_features.npy")
    y_test = np.load(ROOT / "data/test_target.npy")
    train_raw, test_raw = pd.read_csv(ROOT / "data/train.csv"), pd.read_csv(ROOT / "data/test.csv")

    S_train, S_test = model_space(export_pipe, Z_train), model_space(export_pipe, Z_test)
    result = build_detectors(S_train, S_test)
    rep = result["report"]
    cand = {k: (rep[k]["fallback_share_test"], -rep[k]["synthetic_ood_detected_share"]) for k in ("kde", "gmm")}
    selected = min(cand, key=cand.get)
    if abs(rep["kde"]["fallback_share_test"] - rep["gmm"]["fallback_share_test"]) < 0.005:
        selected = max(("kde", "gmm"), key=lambda k: rep[k]["synthetic_ood_detected_share"])
    det = result["detectors"][selected]

    P_train = det.proj.transform(S_train)
    gmm = fit_segments(P_train)
    seg_train = gmm.predict(P_train)
    features = [c for cols in cfg["features"].values() for c in cols]
    numeric = cfg["features"]["numeric_continuous"] + cfg["features"]["numeric_discrete"]
    categorical = cfg["features"]["categorical_nominal"] + cfg["features"]["categorical_ordinal"]
    profiles = profile_segments(train_raw[features], seg_train, numeric, categorical)
    names_path = ROOT / "config/segment_names.json"
    names = json.loads(names_path.read_text(encoding="utf-8")) if names_path.exists() else {}
    save_detectors(result, selected, extra={"segments": {"n_components": int(gmm.n_components), "bic": float(gmm.bic(P_train)),
                                                         "sizes_train": {int(k): int(v) for k, v in zip(*np.unique(seg_train, return_counts=True))},
                                                         "profiles": {str(k): v for k, v in profiles.items()}}})
    save_segmenter(gmm, names)

    from src.app.schemas import PredictionRequest
    from src.features import load_preprocessor
    prep = load_preprocessor()
    candidates = [
        {"duration": 120, "credit_amount": 50000.0, "age": 19, "installment_rate": 4, "residence_since": 1, "existing_credits": 10,
         "checking_account": ">=200 DM", "credit_history": "no credits / all paid", "purpose": "vacation", "savings_account": ">=1000 DM",
         "other_debtors": "guarantor", "property": "unknown/none", "other_installment_plans": "stores", "housing": "for free",
         "telephone": "yes", "employment_since": "unemployed", "job": "unemployed/unskilled non-resident"},
        {"duration": 6, "credit_amount": 50000.0, "age": 100, "installment_rate": 1, "residence_since": 4, "existing_credits": 10,
         "checking_account": "<0 DM", "credit_history": "critical account", "purpose": "retraining", "savings_account": "unknown/none",
         "other_debtors": "co-applicant", "property": "unknown/none", "other_installment_plans": "bank", "housing": "rent",
         "telephone": "yes", "employment_since": "unemployed", "job": "management/highly qualified"},
    ]
    ood_example, ood_density = None, None
    for c in candidates:
        PredictionRequest.model_validate(c)
        ld = float(det.log_density(model_space(export_pipe, prep.transform(pd.DataFrame([c]))))[0])
        if ld < det.threshold and (ood_density is None or ld < ood_density):
            ood_example, ood_density = c, ld
    assert ood_example is not None, "жоден кандидат не є OOD — підберіть інші значення"
    (OOD_DIR / "ood_example.json").write_text(json.dumps(ood_example, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    T, costs = meta["threshold"], cfg["costs"]
    p = export_pipe.predict_proba(Z_test)[:, 1]
    is_ood = det.is_ood(S_test)
    model_label = (p >= T).astype(int)
    rule_label = np.array([non_ml_rule(r) for r in test_raw[features].to_dict("records")])
    system_label = np.where(is_ood, rule_label, model_label)

    def quality(y, lab, score=None):
        out = {"n": int(len(y)), "f1": float(f1_score(y, lab)) if len(np.unique(y)) > 1 else None,
               "cost_total": float(total_cost(y, lab, costs)), "cost_per_applicant": float(total_cost(y, lab, costs) / max(len(y), 1))}
        if score is not None and len(np.unique(y)) > 1:
            out["roc_auc"] = float(roc_auc_score(y, score))
        return out

    evaluation = {
        "selected_detector": selected, "detectors": rep,
        "model_all_test": quality(y_test, model_label, p),
        "model_non_fallback": quality(y_test[~is_ood], model_label[~is_ood], p[~is_ood]),
        "system_model_plus_rule": quality(y_test, system_label),
        "fallback_share_test": float(is_ood.mean()), "n_fallback": int(is_ood.sum()),
        "ood_subset_model_vs_rule": {"n": int(is_ood.sum()),
                                     "model": quality(y_test[is_ood], model_label[is_ood]) if is_ood.sum() else None,
                                     "rule": quality(y_test[is_ood], rule_label[is_ood]) if is_ood.sum() else None,
                                     "note": "висновків не робимо, якщо n < 10"},
        "rule_all_test": quality(y_test, rule_label),
        "ood_example_log_density": ood_density, "threshold": det.threshold,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/ood_evaluation.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    meta.update({"ood_threshold": det.threshold, "ood_detector": selected,
                 "ood_projection": rep["projection"], "segments_n": int(gmm.n_components)})
    (ROOT / "models/model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: evaluation[k] for k in ("selected_detector", "detectors", "fallback_share_test", "model_all_test",
                                                  "system_model_plus_rule", "rule_all_test")}, indent=2, ensure_ascii=False))
    print("сегменти:", evaluation and {int(k): int(v) for k, v in zip(*np.unique(seg_train, return_counts=True))})
    for k, rows in profiles.items():
        print(f"  сегмент {k}:", "; ".join(f"{r['feature']}={r['value']:+.2f}" for r in rows))


if __name__ == "__main__":
    main()
