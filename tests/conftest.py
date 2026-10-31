import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import skops.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

os.environ.setdefault("MODEL_SOURCE", "local")


@pytest.fixture(scope="session")
def meta():
    return json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def export_pipe():
    types = sio.get_untrusted_types(file=str(ROOT / "models" / "pipeline.skops"))
    assert all(t.startswith(("sklearn.", "numpy.")) for t in types), types
    return sio.load(str(ROOT / "models" / "pipeline.skops"), trusted=types)


@pytest.fixture(scope="session")
def X_test():
    return np.load(ROOT / "data" / "test_features.npy").astype(np.float32)


@pytest.fixture(scope="session")
def y_test():
    return np.load(ROOT / "data" / "test_target.npy")


@pytest.fixture(scope="session")
def preprocessor():
    from src.features import load_preprocessor

    return load_preprocessor()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "feedback.db"
    monkeypatch.setenv("FEEDBACK_DB", str(db))
    return db


@pytest.fixture(scope="session")
def valid_payload():
    return {
        "duration": 12, "credit_amount": 2000.0, "age": 35, "installment_rate": 3, "residence_since": 2,
        "existing_credits": 1, "checking_account": "no account", "credit_history": "existing paid duly",
        "purpose": "radio/television", "savings_account": "100-500 DM", "other_debtors": "none",
        "property": "real estate", "other_installment_plans": "none", "housing": "own", "telephone": "none",
        "employment_since": "4-7 years", "job": "skilled employee",
    }


@pytest.fixture(scope="session")
def ood_payload():
    return json.loads((ROOT / "models" / "ood" / "ood_example.json").read_text(encoding="utf-8"))
