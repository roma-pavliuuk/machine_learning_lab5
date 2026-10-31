import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import dvc.api
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
LAB3 = ROOT.parent / "lab_03"
sys.path.insert(0, str(ROOT))


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "config_rf.yaml").read_text(encoding="utf-8"))
    features = [c for cols in cfg["features"].values() for c in cols]
    cols = features + cfg["protected_attributes"] + [cfg["target"]]

    src_md5 = yaml.safe_load((LAB3 / "data/raw/lab2_feature_matrix.csv.dvc").read_text())["outs"][0]["md5"]
    full = pd.read_csv(io.StringIO(dvc.api.read("data/raw/lab2_feature_matrix.csv", repo=str(LAB3))))
    test_idx = pd.read_csv(io.StringIO(dvc.api.read("data/raw/lab2_test_predictions.csv", repo=str(LAB3))),
                           index_col=0).index.to_numpy()
    is_test = full.index.isin(test_idx)
    (ROOT / "data").mkdir(exist_ok=True)
    train, test = full.loc[~is_test, cols], full.loc[is_test, cols]
    train.to_csv(ROOT / "data/train.csv", index=False)
    test.to_csv(ROOT / "data/test.csv", index=False)

    dvc_bin = str(Path(sys.executable).with_name("dvc"))
    if not (ROOT / ".dvc").exists():
        subprocess.run([dvc_bin, "init", "--no-scm", "-q"], cwd=ROOT, check=True)
    subprocess.run([dvc_bin, "add", "data/train.csv", "data/test.csv", "-q"], cwd=ROOT, check=True)
    manifest = {
        "source": {"repo": "lab_03", "path": "data/raw/lab2_feature_matrix.csv", "md5": src_md5,
                   "rev": None, "note": "rev → git-тег після git init"},
        "split": {"origin": "ЛР №2, train_test_split(test_size=0.3, stratify=y, random_state=42)",
                  "n_train": int(len(train)), "n_test": int(len(test)),
                  "bad_share_train": float(train[cfg["target"]].mean()), "bad_share_test": float(test[cfg["target"]].mean())},
        "columns": {"features": features, "protected_for_audit_only": cfg["protected_attributes"], "target": cfg["target"]},
        "md5": {f: hashlib.md5((ROOT / "data" / f).read_bytes()).hexdigest() for f in ("train.csv", "test.csv")},
    }
    (ROOT / "data/manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
