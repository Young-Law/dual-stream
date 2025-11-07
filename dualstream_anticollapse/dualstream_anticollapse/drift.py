
from typing import Tuple
import numpy as np

def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e_hist, _ = np.histogram(expected, bins=cuts)
    a_hist, _ = np.histogram(actual, bins=cuts)
    e_pct = np.where(e_hist == 0, 1e-6, e_hist / max(1, e_hist.sum()))
    a_pct = np.where(a_hist == 0, 1e-6, a_hist / max(1, a_hist.sum()))
    psi = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    return float(psi)

def ks_test(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    expected = expected[np.isfinite(expected)]; actual = actual[np.isfinite(actual)]
    if len(expected)==0 or len(actual)==0:
        return 0.0, 1.0
    try:
        from scipy.stats import ks_2samp
        res = ks_2samp(expected, actual, alternative='two-sided', mode='auto')
        return float(res.statistic), float(res.pvalue)
    except Exception:
        # Fallback: manual KS statistic and asymptotic p-value (approximate)
        data = np.concatenate([expected, actual])
        ecdf_e = np.searchsorted(np.sort(expected), data, side='right')/len(expected)
        ecdf_a = np.searchsorted(np.sort(actual), data, side='right')/len(actual)
        d = float(np.max(np.abs(ecdf_e - ecdf_a)))
        n1, n2 = len(expected), len(actual)
        en = np.sqrt(n1*n2/(n1+n2))
        # Kolmogorov distribution tail approximation
        p = 2*np.exp(-2*(d*en)**2)
        return d, float(max(min(p,1.0),0.0))

class PageHinkley:
    """Concept drift detector (mean shift)"""
    def __init__(self, delta=0.005, lambda_=50, alpha=1.0):
        self.delta = delta; self.lambda_ = lambda_; self.alpha = alpha
        self.reset()
    def reset(self):
        self.mean = 0.0; self.cum = 0.0; self.mincum = 0.0; self.t = 0; self.change = False
    def update(self, x: float) -> bool:
        self.t += 1
        self.mean = self.mean + (x - self.mean)/self.t
        self.cum = self.alpha*(self.cum + x - self.mean - self.delta)
        self.mincum = min(self.mincum, self.cum)
        self.change = (self.cum - self.mincum) > self.lambda_
        return self.change
