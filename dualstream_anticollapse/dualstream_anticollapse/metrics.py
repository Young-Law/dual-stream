
from typing import Dict, Any, Tuple
import numpy as np

def _confusion(y_true, y_pred) -> Tuple[int, int, int, int]:
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    tp = int(np.sum((y_true==1) & (y_pred==1)))
    tn = int(np.sum((y_true==0) & (y_pred==0)))
    fp = int(np.sum((y_true==0) & (y_pred==1)))
    fn = int(np.sum((y_true==1) & (y_pred==0)))
    return tp, tn, fp, fn

def classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    tp, tn, fp, fn = _confusion(y_true, y_pred)
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    res = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp],[fn, tp]],
    }
    if y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score, log_loss
            res["auc"] = float(roc_auc_score(y_true, y_proba))
            # Use probabilities for positive class in log_loss
            eps = 1e-9
            p = np.clip(y_proba, eps, 1 - eps)
            res["log_loss"] = float(log_loss(y_true, p))
        except Exception:
            # simple AUC via rank (Mann–Whitney U)
            y_true = np.array(y_true).astype(int)
            order = np.argsort(y_proba)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(y_proba))
            n1 = np.sum(y_true==1); n0 = np.sum(y_true==0)
            if n1>0 and n0>0:
                u = np.sum(ranks[y_true==1]) - n1*(n1-1)/2
                res["auc"] = float(u / (n0*n1))
    return res
