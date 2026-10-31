import importlib.util
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[2]
LAB3_DIR = Path(os.environ.get("LAB3_DIR", _HERE.parent / "lab_03"))


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    if not path.exists():
        raise FileNotFoundError(f"Не знайдено {path}: покладіть lab_01–lab_03 поруч із lab_05 або задайте LAB3_DIR")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lab3_preprocessor = _load("lab03_preprocessor", LAB3_DIR / "src" / "preprocessor.py")
sys.modules.setdefault("src.preprocessor", lab3_preprocessor)
lab3_evaluation = _load("lab03_evaluation", LAB3_DIR / "src" / "evaluation.py")

DynamicPipelineBuilder = lab3_preprocessor.DynamicPipelineBuilder
build_hermetic_pipeline = lab3_evaluation.build_hermetic_pipeline
load_config = lab3_evaluation.load_config
suggest = lab3_evaluation.suggest
optimal_threshold = lab3_evaluation.optimal_threshold
total_cost = lab3_evaluation.total_cost
cost_curve = lab3_evaluation.cost_curve
SELECT_STEPS = lab3_evaluation.SELECT_STEPS
