
import os, json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from .metrics import classification_metrics
from .drift import population_stability_index, ks_test, PageHinkley
from .alerts import emit
from .coherence import CoherenceAuditor

@dataclass
class MonitorState:
    batches_seen: int = 0
    last_retrain_batch: int = -1
    events: List[Dict[str, Any]] = None

from .edge import zscore_outliers

class ModelMonitor:
    def __init__(self, cfg, baseline_stats: Dict[str, Any], state_path: str, alert_sink: Optional[str]="stdout"):
        self.cfg = cfg
        self.baseline = baseline_stats
        self.state_path = state_path
        self.alert_sink = alert_sink
        self.state = MonitorState(batches_seen=0, last_retrain_batch=-1, events=[])
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                self.state = MonitorState(**json.load(f))

        self.ph = PageHinkley()

    def save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(asdict(self.state), f, indent=2)

    def _feature_cols(self, df: pd.DataFrame):
        exc = [c for c in [self.cfg.target, self.cfg.id_column] if c]
        return self.cfg.features or [c for c in df.columns if c not in exc]

    def check_performance(self, metrics: Dict[str, Any]) -> bool:
        th = self.cfg.thresholds
        base = self.baseline.get("metrics", {})
        triggers = []
        for key, drop_key in [("accuracy","accuracy_drop"), ("f1","f1_drop"), ("auc","auc_drop")]:
            if key in metrics and key in base:
                if metrics[key] < base[key] - getattr(th, drop_key):
                    triggers.append(f"{key}_drop")
        if triggers:
            emit("performance_degradation", {"triggers": triggers, "metrics": metrics, "baseline": base}, sink=self.alert_sink)
            self.state.events.append({"type":"perf", "triggers":triggers, "metrics":metrics})
            return True
        return False

    def check_drift(self, ref: pd.DataFrame, cur: pd.DataFrame) -> bool:
        th = self.cfg.thresholds
        feats = self._feature_cols(ref)
        drifted = []
        for col in feats:
            if not np.issubdtype(ref[col].dtype, np.number): 
                continue
            psi = population_stability_index(ref[col].values, cur[col].values)
            ksD, p = ks_test(ref[col].values, cur[col].values)
            if psi >= th.psi or p < th.ks_pvalue:
                drifted.append({"feature": col, "psi": float(psi), "ks_pvalue": float(p)})
        if drifted:
            emit("data_drift", {"drifted": drifted}, sink=self.alert_sink)
            self.state.events.append({"type":"drift", "details":drifted})
            return True
        return False

    def check_concept_drift(self, y_losses: List[float]) -> bool:
        changed = any(self.ph.update(float(l)) for l in y_losses)
        if changed:
            emit("concept_drift", {"message": "Page-Hinkley triggered"}, sink=self.alert_sink)
            self.state.events.append({"type":"concept_drift"})
        return changed

    def audit_dual_streams(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        auditor = CoherenceAuditor(thresholds={
            "max_allowed_deception_tokens": self.cfg.thresholds.max_allowed_deception_tokens,
            "max_allowed_conflict_markers": self.cfg.thresholds.max_allowed_conflict_markers,
        })
        results = []
        for rec in records:
            out = auditor.audit_record(rec)
            if not out["coherent"]:
                emit("coherence_violation", out, sink=self.alert_sink)
            results.append(out)
        return results

    def check_outliers(self, df):
        feats = self._feature_cols(df)
        out = zscore_outliers(df, feats, z=3.5)
        if out:
            from .alerts import emit
            emit(
                "outliers_detected",
                {"columns": list(out.keys()), "counts": {k: len(v) for k, v in out.items()}},
                sink=self.alert_sink,
            )
            self.state.events.append({"type": "outliers", "details": {k: len(v) for k, v in out.items()}})
            return True, out
        return False, {}
