import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
import yaml
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.app.ood import load_detector, model_space, non_ml_rule


def dpr_eor(granted, good, group, privileged):
    priv = group == privileged
    out = {}
    for name, m in (("privileged", priv), ("unprivileged", ~priv)):
        out[name] = {"n": int(m.sum()), "n_good": int((m & good).sum()), "grant_rate": float(granted[m].mean()) if m.any() else None,
                     "tpr_good": float(granted[m & good].mean()) if (m & good).any() else None}
    p, u = out["privileged"], out["unprivileged"]
    out["DPR"] = u["grant_rate"] / p["grant_rate"] if p["grant_rate"] else None
    out["EOR"] = (u["tpr_good"] / p["tpr_good"]) if (p["tpr_good"] and u["tpr_good"] is not None) else None
    return out


def bootstrap(fn, n, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            v = fn(i)
            if v is not None and np.isfinite(v):
                vals.append(v)
        except ZeroDivisionError:
            pass
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def cramers_v(a, b):
    t = pd.crosstab(a, b)
    chi2 = chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.to_numpy().sum() * (min(t.shape) - 1))))


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config/config_rf.yaml").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "models/model_meta.json").read_text(encoding="utf-8"))
    stamp = {"run_id": meta.get("run_id"), "model_version": meta.get("release"), "threshold": meta["threshold"]}
    pipe_path = ROOT / "models/pipeline.skops"
    pipe = sio.load(str(pipe_path), trusted=sio.get_untrusted_types(file=str(pipe_path)))
    det = load_detector()
    test = pd.read_csv(ROOT / "data/test.csv")
    Z_test, y = np.load(ROOT / "data/test_features.npy"), np.load(ROOT / "data/test_target.npy")
    features = [c for cols in cfg["features"].values() for c in cols]
    attr, priv = cfg["fairness"]["attribute"], cfg["fairness"]["privileged"]
    group = test[attr].to_numpy()

    p = pipe.predict_proba(Z_test)[:, 1]
    is_ood = det.is_ood(model_space(pipe, Z_test))
    model_label = (p >= meta["threshold"]).astype(int)
    rule_label = np.array([non_ml_rule(r) for r in test[features].to_dict("records")])
    system_label = np.where(is_ood, rule_label, model_label)
    good = y == 0

    fairness = {**stamp, "attribute": attr, "privileged": priv, "favourable_decision": "label = 0 (видати кредит)"}
    for name, lab in (("system", system_label), ("model", model_label), ("rule", rule_label)):
        granted = lab == 0
        res = dpr_eor(granted, good, group, priv)
        for g in ("privileged", "unprivileged"):
            res[g]["sufficient_data"] = res[g]["n"] >= 30 and res[g]["n_good"] >= 10
        if all(res[g]["sufficient_data"] for g in ("privileged", "unprivileged")):
            res["DPR_ci"] = bootstrap(lambda i: dpr_eor(granted[i], good[i], group[i], priv)["DPR"], len(y))
            res["EOR_ci"] = bootstrap(lambda i: dpr_eor(granted[i], good[i], group[i], priv)["EOR"], len(y))
            res["DPR_four_fifths"] = res["DPR"] >= 0.8 if res["DPR"] is not None else None
            res["DPR_ci_contains_1"] = (res["DPR_ci"][0] <= 1 <= res["DPR_ci"][1]) if res["DPR_ci"] else None
        fairness[name] = res
    fairness["positives_by_group"] = {g: int(((group == g) & (y == 1)).sum()) for g in np.unique(group)}
    female = (group != priv).astype(int)
    proxies = {}
    for col in features:
        s = test[col]
        if s.dtype.kind in "if" and s.nunique() > 6:
            m = s.notna().to_numpy()
            proxies[col] = {"metric": "|2·AUC−1|", "value": float(abs(roc_auc_score(female[m], s[m]) - 0.5) * 2)}
        else:
            proxies[col] = {"metric": "V Крамера", "value": cramers_v(s.astype("string").fillna("Missing"), female)}
    fairness["proxy_association_with_sex"] = dict(sorted(proxies.items(), key=lambda kv: -kv[1]["value"]))
    (ROOT / "reports/fairness_test.json").write_text(json.dumps(fairness, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    c = cfg["costs"]
    scenarios = {"песимістичний": {"C01": c["C01"] * 1.5, "C10": c["C10"] * 0.6},
                 "центральний": {"C01": c["C01"], "C10": c["C10"]},
                 "оптимістичний": {"C01": c["C01"] * 0.7, "C10": c["C10"] * 1.3}}

    def confusion(lab):
        return {"TP": int(((y == 1) & (lab == 1)).sum()), "FP": int(((y == 0) & (lab == 1)).sum()),
                "FN": int(((y == 1) & (lab == 0)).sum()), "TN": int(((y == 0) & (lab == 0)).sum())}

    def loss(lab, C01, C10, idx=None):
        yy, ll = (y, lab) if idx is None else (y[idx], lab[idx])
        return float(C10 * ((yy == 1) & (ll == 0)).sum() + C01 * ((yy == 0) & (ll == 1)).sum())

    roi = {**stamp, "currency": "DM (Deutsche Mark, 1994)", "n_test": int(len(y)), "cost_matrix_lab3": {"C01_FP": c["C01"], "C10_FN": c["C10"]},
           "assumptions": {"C01": "втрачена чиста процентна маржа медіанного кредиту (2 320 DM × 4.5 % × 1.5 року), ЛР №3",
                           "C10": "LGD: середній кредит 3 271 DM × EAD 75 % × LGD 60 %, ЛР №3",
                           "scenarios": "песимістичний: C01 ×1.5, C10 ×0.6; оптимістичний: C01 ×0.7, C10 ×1.3"},
           "baseline": "Non-ML правило ЛР №1 на тому самому test", "confusion": {"system": confusion(system_label), "rule": confusion(rule_label), "model": confusion(model_label)},
           "fallback_share": float(is_ood.mean()), "scenarios": {}}
    for name, cc in scenarios.items():
        l_rule, l_sys = loss(rule_label, **cc), loss(system_label, **cc)
        entry = {"costs": cc, "loss_rule": l_rule, "loss_system": l_sys, "effect": l_rule - l_sys, "effect_per_applicant": (l_rule - l_sys) / len(y)}
        if name == "центральний":
            entry["effect_ci95"] = bootstrap(lambda i: loss(rule_label, **cc, idx=i) - loss(system_label, **cc, idx=i), len(y))
        roi["scenarios"][name] = entry
    (ROOT / "reports/roi.json").write_text(json.dumps(roi, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"system": {k: fairness["system"][k] for k in ("DPR", "EOR", "DPR_ci", "EOR_ci") if k in fairness["system"]},
                      "model": {k: fairness["model"][k] for k in ("DPR", "EOR")}, "roi": roi["scenarios"], "confusion": roi["confusion"]},
                     indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
