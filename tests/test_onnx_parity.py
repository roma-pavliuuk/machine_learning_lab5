import itertools

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from src.app.schemas import CATEGORY_LEVELS, NUMERIC_BOUNDS, PredictionRequest

from conftest import ROOT


@pytest.fixture(scope="module")
def sess():
    return ort.InferenceSession(str(ROOT / "models" / "pipeline.onnx"), providers=["CPUExecutionProvider"])


def _atol(meta):
    return 1e-4 if meta["model_family"] == "tree_ensemble" else 1e-5


def test_parity_on_test(export_pipe, X_test, meta, sess):
    p_onnx = sess.run(None, {meta["onnx_inputs"][0]["name"]: X_test})[1]
    p_skl = export_pipe.predict_proba(X_test)
    np.testing.assert_allclose(p_skl, p_onnx, atol=_atol(meta))


def test_label_parity_after_threshold(export_pipe, X_test, meta, sess):
    T, atol = meta["threshold"], _atol(meta)
    p_onnx = sess.run(None, {meta["onnx_inputs"][0]["name"]: X_test})[1][:, 1]
    p_skl = export_pipe.predict_proba(X_test)[:, 1]
    flipped = (p_skl >= T) != (p_onnx >= T)
    assert np.all(np.abs(p_skl[flipped] - T) < atol)


def _synthetic_rows(valid_payload):
    rows = []
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        for v in (lo, hi):
            rows.append({**valid_payload, col: v})
    for col, levels in CATEGORY_LEVELS.items():
        for lvl in levels:
            rows.append({**valid_payload, col: lvl})
    for d, a in itertools.product(NUMERIC_BOUNDS["duration"], NUMERIC_BOUNDS["credit_amount"]):
        rows.append({**valid_payload, "duration": d, "credit_amount": a})
    return rows


def test_parity_on_boundaries_and_categories(export_pipe, preprocessor, meta, sess, valid_payload):
    candidates = _synthetic_rows(valid_payload)
    rows = []
    for r in candidates:
        try:
            PredictionRequest.model_validate(r); rows.append(r)
        except ValueError:
            pass
    assert len(rows) >= len(candidates) - 4
    Z = preprocessor.transform(pd.DataFrame(rows))
    assert Z.shape == (len(rows), len(meta["features"])) and np.isfinite(Z).all()
    p_onnx = sess.run(None, {meta["onnx_inputs"][0]["name"]: Z})[1]
    np.testing.assert_allclose(export_pipe.predict_proba(Z), p_onnx, atol=_atol(meta))


def test_meta_contract_matches_artifacts(meta, sess, preprocessor):
    assert [i.name for i in sess.get_inputs()] == [i["name"] for i in meta["onnx_inputs"]]
    assert sess.get_inputs()[0].shape[-1] == len(meta["features"]) == len(preprocessor.feature_names)
    assert preprocessor.feature_names == meta["features"]
    assert set(PredictionRequest.model_fields) == set(meta["request_fields"])
    assert meta["classes"] == [0, 1] and meta["positive_class"] == 1 and 0 < meta["threshold"] < 1
