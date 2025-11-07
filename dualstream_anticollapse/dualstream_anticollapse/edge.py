
from typing import Dict, Any, List
import numpy as np
import pandas as pd

def zscore_outliers(df: pd.DataFrame, cols: List[str], z: float = 3.5) -> Dict[str, List[int]]:
    """Return indices of rows that are outliers per column by absolute z-score > z."""
    out = {}
    for c in cols:
        if not np.issubdtype(df[c].dtype, np.number): 
            continue
        vals = df[c].astype(float).values
        mu = np.nanmean(vals); sigma = np.nanstd(vals) or 1.0
        zabs = np.abs((vals - mu) / sigma)
        idx = np.where(zabs > z)[0].tolist()
        if idx:
            out[c] = idx
    return out
