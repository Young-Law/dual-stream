
from typing import Tuple, Dict, Any
import numpy as np

def build_model(kind: str):
    if kind == "random_forest":
        try:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42)
        except Exception as e:
            raise RuntimeError("RandomForestClassifier requires scikit-learn") from e
    # default: SGDClassifier supports partial_fit
    try:
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(loss="log_loss", random_state=42)
    except Exception as e:
        raise RuntimeError("SGDClassifier requires scikit-learn") from e

def fit_model(model, X, y):
    import numpy as np
    y = np.asarray(y).astype(int)
    if hasattr(model, "partial_fit"):
        classes = np.unique(y)
        model.partial_fit(X, y, classes=classes)
    else:
        model.fit(X, y)
    return model

def predict(model, X):
    import numpy as np
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:,1]
    else:
        # decision_function -> sigmoid fallback
        raw = model.decision_function(X)
        proba = 1/(1+np.exp(-raw))
    pred = (proba >= 0.5).astype(int)
    return pred, proba
