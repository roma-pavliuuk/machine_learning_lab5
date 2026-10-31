from __future__ import annotations

import json
import logging
import math
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.app import db
from src.app.ood import OODDetector, load_detector, model_space, non_ml_rule
from src.app.schemas import (CATEGORY_LEVELS, BatchRequest, BatchResponse, ErrorResponse, ExplainRequest, FeedbackRequest,
                             Meta, PredictionRequest, PredictionResponse)
from src.app.segments import Segmenter, load_segmenter

ROOT = Path(__file__).resolve().parents[2]
NAME = "TrustScoreCredit"
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
log = logging.getLogger("trustscore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class AppException(Exception):
    def __init__(self, status_code: int, error_code: str, message: str, request_id: Optional[str] = None):
        self.status_code, self.error_code, self.message, self.request_id = status_code, error_code, message, request_id


def resolve_artifacts() -> Dict:
    source = os.environ.get("MODEL_SOURCE", "registry")
    if source == "local":
        return {"dir": ROOT / "models", "onnx": ROOT / "models" / "pipeline.onnx", "version": "local", "run_id": None}
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI))
    client = MlflowClient()
    mv = client.get_model_version_by_alias(NAME, "champion")
    cache = ROOT / ".mlflow_cache" / mv.run_id
    onnx_dir = cache / "onnx_model"
    if not (cache / "model_meta.json").exists() or not any(onnx_dir.glob("*.onnx")):
        shutil.rmtree(cache, ignore_errors=True)
        mlflow.artifacts.download_artifacts(run_id=mv.run_id, dst_path=str(cache))
        mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=str(onnx_dir))
    onnx_file = next(onnx_dir.rglob("*.onnx"))
    return {"dir": cache, "onnx": onnx_file, "version": f"{NAME}@champion=v{mv.version}", "run_id": mv.run_id,
            "sklearn_dir": cache / "sklearn", "ood_dir": cache / "ood"}


def check_contract(meta: Dict, sess: ort.InferenceSession, request_model, run_id: Optional[str] = None) -> List[str]:
    errs = []
    if set(request_model.model_fields) != set(meta["request_fields"]):
        errs.append("поля PredictionRequest не збігаються з request_fields")
    inputs = sess.get_inputs()
    if [i.name for i in inputs] != [i["name"] for i in meta["onnx_inputs"]]:
        errs.append("входи ONNX не збігаються з onnx_inputs")
    if len(inputs) == 1 and isinstance(inputs[0].shape[-1], int) and inputs[0].shape[-1] != len(meta["features"]):
        errs.append("ширина входу ONNX не збігається з features")
    if run_id and meta.get("run_id") != run_id:
        errs.append("model_meta.json з іншого run")
    return errs


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready, app.state.errors = False, []
    try:
        art = resolve_artifacts()
        meta_path = art["dir"] / "model_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sess = ort.InferenceSession(str(art["onnx"]), providers=["CPUExecutionProvider"])
        errs = check_contract(meta, sess, PredictionRequest, art["run_id"])
        if errs:
            raise RuntimeError("; ".join(errs))
        if len(meta["features"]) != len(meta["dtypes"]):
            raise RuntimeError("features і dtypes різної довжини")

        import skops.io as sio

        from src.features import load_preprocessor
        skl_dir = art.get("sklearn_dir", art["dir"])
        prep = load_preprocessor(skl_dir / "preprocessor.skops")
        if prep.feature_names != meta["features"]:
            raise RuntimeError("порядок ознак препроцесора не збігається з features у model_meta.json")
        pipe_path = skl_dir / "pipeline.skops"
        export_pipe = sio.load(str(pipe_path), trusted=sio.get_untrusted_types(file=str(pipe_path)))
        background = np.load(skl_dir / "shap_background.npy")

        ood_dir = art.get("ood_dir", art["dir"] / "ood")
        detector: Optional[OODDetector] = None
        segmenter: Optional[Segmenter] = None
        if (ood_dir / "ood_meta.json").exists():
            detector = load_detector(ood_dir)
            if (ood_dir / "segments_gmm.skops").exists():
                segmenter = load_segmenter(detector.proj, ood_dir)
            if "ood_threshold" in meta and not math.isclose(meta["ood_threshold"], detector.threshold, rel_tol=1e-9):
                raise RuntimeError("ood_threshold у model_meta.json не збігається з детектором")
        else:
            log.warning("OOD-компонентів немає — fallback вимкнено (етап 3)")

        from src.xai.explain import LocalExplainer
        categorical = list(CATEGORY_LEVELS)
        explainer = LocalExplainer(export_pipe, background, meta["features"], meta["request_fields"], categorical)

        db.init_db()
        for key, value in dict(meta=meta, sess=sess, input_name=meta["onnx_inputs"][0]["name"], prep=prep, export_pipe=export_pipe,
                               detector=detector, segmenter=segmenter, explainer=explainer, model_version=art["version"],
                               threshold=float(meta["threshold"]), positive=meta["classes"].index(meta["positive_class"])).items():
            setattr(app.state, key, value)
        app.state.ready = True
        log.info("модель %s завантажено; OOD: %s; сегменти: %s", art["version"], detector.kind if detector else "—",
                 bool(segmenter))
    except Exception as exc:
        app.state.errors.append(str(exc))
        log.error("lifespan: сервіс не готовий: %s", exc)
    yield


app = FastAPI(title="TrustScore AI — кредитний скоринг (ЛР №5)", version="2.0", lifespan=lifespan)


@app.exception_handler(AppException)
def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(status_code=exc.status_code,
                        content=ErrorResponse(error_code=exc.error_code, message=exc.message, request_id=exc.request_id).model_dump())


@app.exception_handler(RequestValidationError)
def validation_handler(request: Request, exc: RequestValidationError):
    msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])
    return JSONResponse(status_code=422, content=ErrorResponse(error_code="validation_error", message=msg).model_dump())


@app.exception_handler(StarletteHTTPException)
def http_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(error_code=f"http_{exc.status_code}", message=str(exc.detail)).model_dump())


def _require_ready() -> None:
    if not app.state.ready:
        raise AppException(503, "model_not_ready", "; ".join(app.state.errors) or "lifespan ще не завершено")


def _entropy(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(-(p * math.log(p) + (1 - p) * math.log(1 - p)))


def predict_rows(records: List[Dict]) -> List[PredictionResponse]:
    st = app.state
    Z = st.prep.transform(pd.DataFrame(records))
    proba = st.sess.run(None, {st.input_name: Z})[1][:, st.positive]
    if st.detector is not None:
        S = model_space(st.export_pipe, Z)
        logdens = st.detector.log_density(S)
        is_ood = logdens < st.detector.threshold
        seg_ids = st.segmenter.assign(S) if st.segmenter is not None else [None] * len(records)
    else:
        logdens, is_ood, seg_ids = [None] * len(records), [False] * len(records), [None] * len(records)
    out = []
    for rec, p, ld, ood, seg in zip(records, proba, logdens, is_ood, seg_ids):
        rid = str(uuid.uuid4())
        p = float(p)
        label = int(non_ml_rule(rec)) if ood else int(p >= st.threshold)
        segment = None if (ood or seg is None) else st.segmenter.describe(int(seg))
        meta = Meta(model_version=st.model_version, fallback_applied=bool(ood),
                    ood_log_density=None if ld is None else float(ld), segment=segment)
        resp = PredictionResponse(request_id=rid, score=p, label=label, uncertainty=_entropy(p), meta=meta)
        db.log_prediction(rid, rec, p, label, resp.uncertainty, st.model_version, bool(ood),
                          meta.ood_log_density, None if segment is None else segment["id"])
        out.append(resp)
    return out


@app.get("/health")
def health():
    if not app.state.ready:
        return JSONResponse(status_code=503, content={"status": "not_ready", "errors": app.state.errors})
    st = app.state
    return {"status": "ok", "model_version": st.model_version, "threshold": st.threshold,
            "ood": st.detector.kind if st.detector else None, "segments": st.segmenter is not None}


@app.post("/v1/predict", response_model=PredictionResponse, responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def predict(req: PredictionRequest):
    _require_ready()
    return predict_rows([req.model_dump()])[0]


@app.post("/v1/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    _require_ready()
    return BatchResponse(results=predict_rows([r.model_dump() for r in req.items]))


@app.post("/v1/feedback")
def feedback(req: FeedbackRequest):
    _require_ready()
    if not db.log_feedback(str(req.request_id), req.y_true):
        raise AppException(404, "unknown_request_id", "request_id не знайдено в журналі прогнозів", str(req.request_id))
    return {"request_id": str(req.request_id), "y_true": req.y_true, "stored": True}


@app.post("/v1/explain")
def explain(req: ExplainRequest):
    _require_ready()
    st = app.state
    record = req.request.model_dump()
    pred = predict_rows([record])[0]
    z = st.prep.transform(pd.DataFrame([record]))
    result = st.explainer.explain(z, record, lime_samples=req.lime_samples)
    result["prediction"] = pred.model_dump(mode="json")
    result["note"] = ("fallback: відповідь дало Non-ML правило; SHAP і LIME описують модель лише довідково"
                      if pred.meta.fallback_applied else None)
    return result
