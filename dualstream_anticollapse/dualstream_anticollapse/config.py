
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Thresholds:
    # performance thresholds
    accuracy_drop: float = 0.05
    f1_drop: float = 0.05
    auc_drop: float = 0.05
    # drift thresholds
    psi: float = 0.2
    ks_pvalue: float = 0.01
    # coherence thresholds
    max_allowed_deception_tokens: int = 0
    max_allowed_conflict_markers: int = 0

@dataclass
class RetrainPolicy:
    kind: str = "triggered"   # "scheduled" | "triggered"
    min_batches_between_retrains: int = 3
    schedule_every_n_batches: int = 10

@dataclass
class Config:
    target: str
    id_column: Optional[str] = None
    features: Optional[List[str]] = None
    model_type: str = "sgd_classifier"  # or "random_forest"
    output_dir: str = "artifacts"
    thresholds: Thresholds = field(default_factory=Thresholds)
    retrain: RetrainPolicy = field(default_factory=RetrainPolicy)
