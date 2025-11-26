
import os, json, hashlib, time, pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union

def _sha256_path(path: Union[str, Path]) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def save_model(model, path: Union[str, Path], meta: Dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump(model, path)
    except Exception:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    record = {"path": str(path), "sha256": _sha256_path(path), "saved_at": time.time(), **meta}
    with open(path.with_suffix(path.suffix + ".meta.json"), "w") as mf:
        json.dump(record, mf, indent=2)
    return record

def load_model(path: Union[str, Path]):
    path = Path(path)
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

@dataclass
class RegistryItem:
    version: str
    path: str
    sha256: str
    created_at: float
    metrics: Dict[str, Any]

class ModelRegistry:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.index_path = os.path.join(root, "registry.json")
        if not os.path.exists(self.index_path):
            with open(self.index_path, "w") as f:
                json.dump({"models": []}, f)

    def add(self, item: RegistryItem):
        with open(self.index_path, "r") as f:
            idx = json.load(f)
        idx["models"].append(asdict(item))
        with open(self.index_path, "w") as f:
            json.dump(idx, f, indent=2)

    def latest(self) -> Optional[RegistryItem]:
        with open(self.index_path, "r") as f:
            idx = json.load(f)
        if not idx["models"]: return None
        last = idx["models"][-1]
        return RegistryItem(**last)
