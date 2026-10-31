from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import skops.io as sio
import yaml
from scipy.stats import spearmanr
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.tree import DecisionTreeRegressor, export_text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.app.ood import load_detector, model_space
from src.app.schemas import CATEGORY_LEVELS, NUMERIC_BOUNDS, PredictionRequest
from src.app.segments import load_segmenter
from src.features import load_preprocessor
from src.xai.explain import LocalExplainer

OUT = ROOT / "reports" / "xai"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def ale_1d(predict, X: np.ndarray, j: int, bins: int = 20):
    x = X[:, j]
    grid = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    effects = []
    for lo, hi in zip(grid[:-1], grid[1:]):
        mask = (x >= lo) & (x <= hi) if hi == grid[-1] else (x >= lo) & (x < hi)
        if not mask.any():
            effects.append(0.0); continue
        Xa, Xb = X[mask].copy(), X[mask].copy()
        Xa[:, j], Xb[:, j] = lo, hi
        effects.append(float(np.mean(predict(Xb) - predict(Xa))))
    ale = np.cumsum(effects)
    ale -= ale.mean()
    return grid[1:], ale


def bootstrap_ci(y, p, lab, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    aucs, f1s = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        aucs.append(roc_auc_score(y[i], p[i])); f1s.append(f1_score(y[i], lab[i]))
    return {"roc_auc": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            "f1": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config/config_rf.yaml").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "models/model_meta.json").read_text(encoding="utf-8"))
    stamp = {"run_id": meta.get("run_id"), "model_version": meta.get("release"), "threshold": meta["threshold"]}
    pipe_path = ROOT / "models/pipeline.skops"
    pipe = sio.load(str(pipe_path), trusted=sio.get_untrusted_types(file=str(pipe_path)))
    prep = load_preprocessor()
    Z_test, y_test = np.load(ROOT / "data/test_features.npy"), np.load(ROOT / "data/test_target.npy")
    Z_train = np.load(ROOT / "data/train_features.npy")
    test_raw = pd.read_csv(ROOT / "data/test.csv")
    background = np.load(ROOT / "models/shap_background.npy")
    categorical = list(CATEGORY_LEVELS)
    explainer = LocalExplainer(pipe, background, meta["features"], meta["request_fields"], categorical)
    model, head = pipe[-1], pipe[:-1]
    S_test = model_space(pipe, Z_test)
    names = explainer.model_feature_names
    p_test = pipe.predict_proba(Z_test)[:, 1]

    sv = np.asarray(explainer.shap.shap_values(S_test))
    sv = sv[:, :, 1] if sv.ndim == 3 else sv
    short = [n.split("__", 1)[1] for n in names]
    plt.figure(figsize=(8, 5)); shap.summary_plot(sv, S_test, feature_names=short, show=False, max_display=12)
    plt.title("SHAP summary (test, TreeExplainer interventional)", loc="left"); plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=130); plt.close()
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)

    rho = pd.DataFrame(S_test, columns=short).corr(method="spearman").abs().to_numpy()
    np.fill_diagonal(rho, 0)
    top2 = [int(i) for i in order[:2]]
    pdp_info = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, j in zip(axes, top2):
        max_rho = float(rho[j].max()); partner = short[int(rho[j].argmax())]
        binary = len(np.unique(S_test[:, j])) <= 2
        if max_rho > 0.7 and not binary:
            grid, ale = ale_1d(lambda X: model.predict_proba(X)[:, 1], S_test, j)
            ax.plot(grid, ale, "o-", color=PALETTE[1], ms=4); ax.set_title(f"ALE: {short[j]} (|ρ|={max_rho:.2f} з {partner})", loc="left", fontsize=9)
            pdp_info[short[j]] = {"method": "ALE", "max_abs_spearman": max_rho, "partner": partner}
        else:
            PartialDependenceDisplay.from_estimator(model, S_test, [j], kind="both", feature_names=short, ax=ax,
                                                    ice_lines_kw={"color": "#c9c8c3", "alpha": 0.3}, pd_line_kw={"color": PALETTE[0], "lw": 2})
            ax.set_title(f"PDP + ICE: {short[j]} (max |ρ|={max_rho:.2f})", loc="left", fontsize=9)
            pdp_info[short[j]] = {"method": "PDP+ICE", "max_abs_spearman": max_rho, "partner": partner}
    fig.tight_layout(); fig.savefig(OUT / "pdp_ice_or_ale.png", dpi=130); plt.close(fig)

    S_train = model_space(pipe, Z_train)
    p_train = pipe.predict_proba(Z_train)[:, 1]
    tree = DecisionTreeRegressor(max_depth=4, random_state=42).fit(S_train, p_train)
    fidelity = float(r2_score(p_test, tree.predict(S_test)))
    (OUT / "surrogate_tree.txt").write_text(export_text(tree, feature_names=short, decimals=3), encoding="utf-8")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(test_raw), 30, replace=False)
    numeric = cfg["features"]["numeric_continuous"] + cfg["features"]["numeric_discrete"]
    std = test_raw[numeric].std()
    rhos, n_valid = [], 0
    for i in idx:
        rec = test_raw.iloc[i][meta["request_fields"]].to_dict()
        rec = {k: (None if pd.isna(v) else (int(v) if k in ("duration", "age", "installment_rate", "residence_since", "existing_credits") else v)) for k, v in rec.items()}
        base = explainer.shap.shap_values(model_space(pipe, prep.transform(pd.DataFrame([rec]))))
        base = np.asarray(base); base = base[0, :, 1] if base.ndim == 3 else base[0]
        top5 = np.argsort(-np.abs(base))[:5]
        for _ in range(20):
            pert = dict(rec)
            for col in ("duration", "credit_amount", "age"):
                if pert[col] is None: continue
                noise = rng.normal(0, rng.uniform(0.01, 0.05) * std[col])
                lo, hi = NUMERIC_BOUNDS[col]
                val = float(np.clip(pert[col] + noise, lo, hi))
                pert[col] = round(val) if col != "credit_amount" else float(round(val))
            try:
                PredictionRequest.model_validate(pert); n_valid += 1
            except Exception:
                continue
            after = explainer.shap.shap_values(model_space(pipe, prep.transform(pd.DataFrame([pert]))))
            after = np.asarray(after); after = after[0, :, 1] if after.ndim == 3 else after[0]
            r = spearmanr(base[top5], after[top5]).statistic
            if r == r:
                rhos.append(float(r))
    stability = {"n_objects": 30, "n_perturbations": 20, "n_valid": n_valid, "mean_spearman_top5": float(np.mean(rhos)),
                 "sd": float(np.std(rhos, ddof=1)), "noise": "гаусів 1–5 % std для duration, credit_amount, age; обрізання до меж Field"}

    det = load_detector(); seg = load_segmenter(det.proj)
    seg_test = seg.assign(S_test); is_ood = det.is_ood(S_test)
    lab = (p_test >= meta["threshold"]).astype(int)
    overall = {"n": int(len(y_test)), "bad_share": float(y_test.mean()), "roc_auc": float(roc_auc_score(y_test, p_test)),
               "f1": float(f1_score(y_test, lab)), **{"ci": bootstrap_ci(y_test, p_test, lab)}}
    segments = []
    for k in np.unique(seg.gmm.predict(det.proj.transform(S_train))):
        m = seg_test == k
        entry = {"id": int(k), "name": seg.describe(int(k))["name"], "n_test": int(m.sum()),
                 "n_train": int((seg.gmm.predict(det.proj.transform(S_train)) == k).sum()),
                 "bad_share": float(y_test[m].mean()) if m.any() else None, "fallback_share": float(is_ood[m].mean()) if m.any() else None,
                 "mean_abs_shap": {short[j]: float(np.abs(sv[m, j]).mean()) for j in order[:5]} if m.any() else {}}
        if m.sum() >= 30 and len(np.unique(y_test[m])) == 2 and y_test[m].sum() >= 10:
            entry.update({"roc_auc": float(roc_auc_score(y_test[m], p_test[m])), "f1": float(f1_score(y_test[m], lab[m]))})
            rng2 = np.random.default_rng(42); diffs = []
            for _ in range(1000):
                i = rng2.integers(0, len(y_test), len(y_test)); mi = m[i]
                if len(np.unique(y_test[i][mi])) < 2 or len(np.unique(y_test[i])) < 2: continue
                diffs.append(roc_auc_score(y_test[i][mi], p_test[i][mi]) - roc_auc_score(y_test[i], p_test[i]))
            lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
            entry.update({"auc_diff_ci": [lo, hi], "limited_trust": hi < 0})
        else:
            entry.update({"roc_auc": None, "f1": None, "note": "недостатньо даних (< 30 об'єктів або < 10 позитивних)", "limited_trust": None})
        segments.append(entry)
    seg_report = {**stamp, "overall": overall, "segments": segments, "projection": meta.get("ood_projection")}
    (ROOT / "reports/segments.json").write_text(json.dumps(seg_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    P_train = det.proj.transform(S_train); seg_train = seg.gmm.predict(P_train)
    np.save(OUT / "train_map.npy", np.column_stack([P_train[:, :2], seg_train]))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for k in np.unique(seg_train):
        mm = seg_train == k
        ax.scatter(P_train[mm, 0], P_train[mm, 1], s=8, color=PALETTE[int(k) % 8], label=seg.describe(int(k))["name"][:48], alpha=0.75, edgecolors="none")
    ax.set_xlabel("PC1 (whiten)"); ax.set_ylabel("PC2 (whiten)"); ax.set_title("Карта даних: train у просторі проєкції OOD, колір — сегмент", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(OUT / "data_map.png", dpi=130); plt.close(fig)

    summary = {**stamp, "explainer": {"type": "TreeExplainer", "feature_perturbation": "interventional", "background_rows": int(len(background)),
                                      "model_output": "probability", "lime_num_samples": 1000},
               "global_importance_mean_abs_shap": {short[j]: float(mean_abs[j]) for j in order},
               "pdp": pdp_info, "surrogate": {"max_depth": 4, "fidelity_r2_test": fidelity}, "stability": stability}
    (OUT / "xai_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("pdp", "surrogate", "stability")}, indent=2, ensure_ascii=False))
    print("сегменти:", [(s["id"], s["n_test"], s.get("roc_auc"), s.get("limited_trust")) for s in segments])


if __name__ == "__main__":
    main()
