from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import onnx
import pandas as pd
import yaml
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
NAME = "TrustScoreCredit"
LAB3 = ROOT.parent / "lab_03"


def lab3_metrics() -> dict:
    from scipy import stats

    df = pd.read_csv(LAB3 / "artifacts" / "three_approaches.csv")
    row = df[(df["модель"] == "RF") & (df["підхід"].str.contains("оптимальний поріг T")) & ~df["підхід"].str.contains("додатково")].iloc[0]

    def parse(s):
        m, sd = s.split("±"); return float(m), float(sd)

    out = {}
    n = 5
    for key, col in (("f1", "F1"), ("roc_auc", "ROC-AUC"), ("cost_loss", "DM / заявку")):
        m, sd = parse(row[col])
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        out.update({key: m, f"{key}_ci_low": m - h, f"{key}_ci_high": m + h})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=["v1", "v2"], required=True)
    args = parser.parse_args()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(NAME)

    meta = json.loads((ROOT / "models/model_meta.json").read_text(encoding="utf-8"))
    info = json.loads((ROOT / "models/training_info.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    dvc_train = yaml.safe_load((ROOT / "data/train.csv.dvc").read_text())["outs"][0]["md5"]

    with mlflow.start_run(run_name=f"release-{args.release}") as run:
        mlflow.log_params({**info["best_params"], "threshold": meta["threshold"], "class_weight_1": meta["class_weight"]["1"],
                           "n_input_features": len(meta["features"]), "n_final_features": len(meta["selected_final"])})
        metrics = lab3_metrics()
        metrics.update({"inner_auc_optuna": info["inner_auc"], "oof_auc_train": info["oof_auc"]})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"dvc_data_hash": dvc_train, "dvc_source_hash_lab3": manifest["source"]["md5"],
                         "model_family": meta["model_family"], "release": args.release, "track": "4C"})
        onx = onnx.load(str(ROOT / "models/pipeline.onnx"))
        model_info = mlflow.onnx.log_model(onx, name="onnx_model", registered_model_name=NAME)
        mlflow.log_artifact(str(ROOT / "models/pipeline.skops"), artifact_path="sklearn")
        mlflow.log_artifact(str(ROOT / "models/preprocessor.skops"), artifact_path="sklearn")
        mlflow.log_artifact(str(ROOT / "models/shap_background.npy"), artifact_path="sklearn")
        if args.release == "v2":
            mlflow.log_artifacts(str(ROOT / "models/ood"), artifact_path="ood")
        meta["run_id"] = run.info.run_id
        meta["release"] = args.release
        (ROOT / "models/model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(ROOT / "models/model_meta.json"))
        mlflow.log_artifact(str(ROOT / "deploy/requirements.txt"))

    client = MlflowClient()
    v = str(model_info.registered_model_version)
    client.set_registered_model_alias(NAME, "champion", v)
    client.set_model_version_tag(NAME, v, "parity_atol", "1e-4")
    client.set_model_version_tag(NAME, v, "release", args.release)
    client.set_model_version_tag(NAME, v, "run_id", run.info.run_id)
    if args.release == "v2":
        previous = [mv for mv in client.search_model_versions(f"name='{NAME}'") if str(mv.version) != str(v)]
        for mv in previous:
            client.set_model_version_tag(NAME, mv.version, "superseded_by", v)
    print(f"{NAME} v{v} ← @champion; run_id={run.info.run_id}; metrics={ {k: round(x, 4) for k, x in metrics.items()} }")


if __name__ == "__main__":
    main()
