from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import skl2onnx
import sklearn
import skops.io as sio
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import Preprocessor
from src.features.lab3_bridge import SELECT_STEPS, load_config

MODELS, DATA = ROOT / "models", ROOT / "data"
OPSET = 17


def selection_indices(pipe) -> list:
    idx = np.arange(pipe.named_steps["prep"].get_feature_names_out().shape[0])
    for name in SELECT_STEPS[:-1]:
        step = pipe.named_steps[name]
        idx = idx[list(step.k_feature_idx_)] if hasattr(step, "k_feature_idx_") else idx[step.get_support()]
    return [int(i) for i in idx]


def main() -> None:
    cfg = load_config(ROOT / "config" / "config_rf.yaml")
    info = json.loads((MODELS / "training_info.json").read_text(encoding="utf-8"))
    full = joblib.load(MODELS / "pipeline_full.joblib")
    request_fields = [c for cols in cfg["features"].values() for c in cols]

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y_train, y_test = train[cfg["target"]].to_numpy(), test[cfg["target"]].to_numpy()

    prep = Preprocessor(full.named_steps["prep"], request_fields)
    Z_train, Z_test = prep.transform(train), prep.transform(test)

    idx = selection_indices(full)
    select = ColumnTransformer([("sel", "passthrough", idx)], remainder="drop").fit(Z_train)
    export_pipe = Pipeline([("select", select),
                            ("embedded", full.named_steps["embedded"]),
                            ("model", full.named_steps["model"])])
    assert not any(hasattr(s, "fit_resample") for _, s in export_pipe.steps), "у графі лишився семплер"

    p_full = full.predict_proba(test.drop(columns=cfg["protected_attributes"] + [cfg["target"]]))[:, 1]
    p_export = export_pipe.predict_proba(Z_test)[:, 1]
    np.testing.assert_allclose(p_full, p_export, atol=1e-6)

    clf = export_pipe.steps[-1][1]
    onx = convert_sklearn(export_pipe, initial_types=[("float_input", FloatTensorType([None, Z_train.shape[1]]))],
                          options={id(clf): {"zipmap": False}}, target_opset=OPSET)
    MODELS.mkdir(exist_ok=True)
    (MODELS / "pipeline.onnx").write_bytes(onx.SerializeToString())

    sio.dump(export_pipe, MODELS / "pipeline.skops")
    sio.dump(prep, MODELS / "preprocessor.skops")
    rng = np.random.default_rng(42)
    bg = rng.choice(len(Z_train), size=min(300, len(Z_train)), replace=False)
    np.save(MODELS / "shap_background.npy", Z_train[bg].astype(np.float32))
    np.save(DATA / "train_features.npy", Z_train); np.save(DATA / "test_features.npy", Z_test)
    np.save(DATA / "train_target.npy", y_train); np.save(DATA / "test_target.npy", y_test)

    sess = ort.InferenceSession(str(MODELS / "pipeline.onnx"), providers=["CPUExecutionProvider"])
    selected = [prep.feature_names[i] for i in idx]
    final = list(np.array(selected)[export_pipe.named_steps["embedded"].get_support()])
    lab1_file = Path(sys.modules["lab03_preprocessor"].MODULES["lab01_dynamic_pipeline_builder"][0])
    meta = {
        "project": cfg.get("project_name", "TrustScoreCredit"),
        "request_fields": request_fields,
        "features": prep.feature_names,
        "dtypes": {f: "float32" for f in prep.feature_names},
        "onnx_inputs": [{"name": i.name, "shape": [d.dim_value or None for d in i.type.tensor_type.shape.dim]}
                        for i in onx.graph.input],
        "onnx_outputs": [o.name for o in sess.get_outputs()],
        "classes": [int(c) for c in clf.classes_],
        "positive_class": 1,
        "threshold": float(info["threshold"]),
        "model_family": "tree_ensemble",
        "opset": OPSET,
        "graph_boundary": {"outside_onnx": "DynamicPipelineBuilder ЛР №1 (AdaptiveImputer, InteractionTransformer, "
                                           "Winsorizer, OneHotEncoder, OrdinalEncoder) — models/preprocessor.skops",
                           "inside_onnx": ["ColumnTransformer(passthrough: CorrelationFilter→SelectKBest→SFS індекси)",
                                           "SelectFromModel(L1-логістична)", "RandomForestClassifier"]},
        "selected_after_sfs": selected, "selected_final": final,
        "best_params": info["best_params"], "class_weight": {str(k): v for k, v in clf.class_weight.items()},
        "preprocessing_version": f"lab_01/src/dynamic_pipeline_builder.py@sha256:{hashlib.sha256(lab1_file.read_bytes()).hexdigest()[:12]}",
        "library_versions": {m.__name__: m.__version__ for m in (sklearn, skl2onnx, ort, np)},
        "data": json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))["md5"],
    }
    (MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    atol = 1e-4
    p_onnx = sess.run(None, {"float_input": Z_test})[1][:, 1]
    T = meta["threshold"]
    flipped = np.flatnonzero((p_export >= T) != (p_onnx >= T))
    near = np.flatnonzero(np.abs(p_export - T) < atol)
    report = {"n_test": int(len(Z_test)), "max_abs_diff": float(np.abs(p_export - p_onnx).max()), "atol": atol,
              "threshold": T, "n_label_changed": int(len(flipped)), "label_changed_rows": flipped.tolist(),
              "rows_within_atol_of_threshold": near.tolist(), "n_input_features": int(Z_train.shape[1]),
              "n_selected_after_sfs": len(selected), "n_final_features": len(final)}
    (MODELS / "parity_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("фінальні ознаки RF:", final)


if __name__ == "__main__":
    main()
