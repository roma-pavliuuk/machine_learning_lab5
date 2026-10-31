import argparse
import json
import os
import sys
import time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.features.lab3_bridge import build_hermetic_pipeline, load_config, optimal_threshold, suggest

CONFIG = ROOT / "config" / "config_rf.yaml"


def _fold_auc(config_path, params, X, y, tr, va):
    sys.path.insert(0, str(ROOT))
    from src.features.lab3_bridge import build_hermetic_pipeline as build

    model = build(config_path, cost_sensitive=True, params=params).fit(X.iloc[tr], y[tr])
    return roc_auc_score(y[va], model.predict_proba(X.iloc[va])[:, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    cfg = load_config(CONFIG)
    train = pd.read_csv(ROOT / "data/train.csv")
    y = train.pop(cfg["target"]).to_numpy()
    X = train.drop(columns=cfg["protected_attributes"])
    inner = StratifiedKFold(**cfg["cv"]["inner"])
    folds = list(inner.split(X, y))
    space = cfg["search"]["space"][cfg["model_family"]]

    def objective(trial):
        params = suggest(trial, space)
        aucs = Parallel(n_jobs=3)(delayed(_fold_auc)(CONFIG, params, X, y, tr, va) for tr, va in folds)
        return float(np.mean(aucs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["search"]["seed"]))
    study.optimize(objective, n_trials=args.trials)
    print(f"Optuna: {args.trials} спроб за {(time.time() - t0) / 60:.1f} хв; best inner AUC = {study.best_value:.4f}; "
          f"params = {study.best_params}", flush=True)

    best = build_hermetic_pipeline(CONFIG, cost_sensitive=True, params=study.best_params)
    oof = cross_val_predict(best, X, y, cv=inner, method="predict_proba")[:, 1]
    threshold = optimal_threshold(y, oof, cfg["costs"])
    best.fit(X, y)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(best, ROOT / "models/pipeline_full.joblib")
    info = {"best_params": study.best_params, "inner_auc": study.best_value, "threshold": threshold,
            "n_trials": args.trials, "inner_cv": cfg["cv"]["inner"], "oof_auc": float(roc_auc_score(y, oof)),
            "trials": study.trials_dataframe(attrs=("number", "value", "params")).to_dict("records")}
    (ROOT / "models/training_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"поріг T (внутрішні OOF) = {threshold:.3f}; OOF AUC = {info['oof_auc']:.4f}; модель збережено", flush=True)


if __name__ == "__main__":
    main()
