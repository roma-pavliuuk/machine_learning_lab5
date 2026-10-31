import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MODEL_SOURCE", "local")
os.environ["FEEDBACK_DB"] = str(ROOT / ".latency_feedback.db")

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.app.main import app


def main() -> None:
    meta = json.loads((ROOT / "models/model_meta.json").read_text(encoding="utf-8"))
    test = pd.read_csv(ROOT / "data/test.csv")
    rows = []
    for rec in test[meta["request_fields"]].head(100).to_dict("records"):
        rows.append({k: (None if pd.isna(v) else (int(v) if k in ("duration", "age", "installment_rate", "residence_since", "existing_credits") else v))
                     for k, v in rec.items()})
    batch = {"items": [r for r in rows] * 3}
    batch["items"] = batch["items"][:256]
    result = {"run_id": meta.get("run_id"), "model_version": meta.get("release"), "n_requests": 100,
              "machine": {"cpu": platform.processor() or os.uname().machine, "cores": os.cpu_count(), "python": platform.python_version(),
                          "os": f"{platform.system()} {platform.release()}"}, "endpoints": {}}
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        for name, call in (("/v1/predict", lambda i: client.post("/v1/predict", json=rows[i % len(rows)])),
                           ("/v1/predict_batch (256)", lambda i: client.post("/v1/predict_batch", json=batch)),
                           ("/v1/explain", lambda i: client.post("/v1/explain", json={"request": rows[i % len(rows)], "lime_samples": 1000}))):
            n = 100 if "explain" not in name else 30
            times = []
            for i in range(n):
                t = time.perf_counter(); r = call(i); times.append((time.perf_counter() - t) * 1000)
                assert r.status_code == 200, r.text
            result["endpoints"][name] = {"n": n, "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
                                        "mean_ms": float(np.mean(times))}
    (ROOT / "reports/latency.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(os.environ["FEEDBACK_DB"]).unlink(missing_ok=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
