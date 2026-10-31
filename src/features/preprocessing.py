from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.skops"


class Preprocessor:
    def __init__(self, prep, request_fields: List[str]):
        self.prep = prep
        self.request_fields = list(request_fields)
        self.feature_names = [str(n) for n in prep.get_feature_names_out()]

    def frame(self, records) -> pd.DataFrame:
        df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records.copy()
        for col in self.request_fields:
            if col not in df.columns:
                df[col] = np.nan
        return df[self.request_fields]

    def transform(self, records) -> np.ndarray:
        Z = self.prep.transform(self.frame(records))
        Z = Z.toarray() if hasattr(Z, "toarray") else Z
        return np.asarray(Z, dtype=np.float32)


def feature_names_of(prep) -> List[str]:
    return [str(n) for n in prep.get_feature_names_out()]


def load_preprocessor(path: Path = PREPROCESSOR_PATH) -> Preprocessor:
    import skops.io as sio

    import src.features.lab3_bridge

    types = sio.get_untrusted_types(file=str(path))
    allowed_prefixes = ("src.features.preprocessing.Preprocessor", "lab01_dynamic_pipeline_builder.", "lab02_", "sklearn.", "numpy.")
    bad = [t for t in types if not t.startswith(allowed_prefixes)]
    if bad:
        raise ValueError(f"Неочікувані типи у skops-артефакті: {bad}")
    return sio.load(str(path), trusted=types)
