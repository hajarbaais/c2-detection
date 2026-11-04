"""P6 - métriques d'évaluation communes (baselines, Isolation Forest, puis LSTM / fusion).

Règle du projet : jamais l'accuracy seule. On rapporte recall et FPR, avec intervalle de
confiance (Wilson 95 %), globalement puis par source (dataset) et par famille.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance de Wilson pour une proportion k/n (n = 0 -> (nan, nan))."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def threshold_at_fpr(benign_scores, target_fpr: float) -> float:
    """Seuil tel que la part de scores bénins STRICTEMENT au-dessus soit <= target_fpr.

    Convention : score élevé = suspect ; alerte si score > seuil.
    """
    s = np.sort(np.asarray(benign_scores, dtype=float)[~np.isnan(benign_scores)])
    if s.size == 0:
        return math.inf
    k = int(math.floor(target_fpr * s.size))      # nb max de bénins autorisés au-dessus
    return float(s[s.size - k - 1])


def binary_metrics(y_true, y_pred, scores=None) -> dict:
    """recall, FPR (+ IC Wilson), précision, F1, ROC-AUC et PR-AUC (si scores fournis)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(bool)
    tp, fn = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum())
    fp, tn = int((p & (y == 0)).sum()), int((~p & (y == 0)).sum())
    n_pos, n_neg = tp + fn, fp + tn
    recall = tp / n_pos if n_pos else math.nan
    fpr = fp / n_neg if n_neg else math.nan
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    out = dict(n_c2=n_pos, n_benign=n_neg, tp=tp, fn=fn, fp=fp, tn=tn,
               recall=recall, fpr=fpr, precision=precision, f1=f1)
    out["recall_ci_low"], out["recall_ci_high"] = wilson_ci(tp, n_pos)
    out["fpr_ci_low"], out["fpr_ci_high"] = wilson_ci(fp, n_neg)
    if scores is not None and n_pos and n_neg:
        sc = np.asarray(scores, dtype=float)
        if np.isnan(sc).any():                       # score manquant = le moins suspect
            fill = np.nanmin(sc) - 1 if (~np.isnan(sc)).any() else 0.0
            sc = np.where(np.isnan(sc), fill, sc)
        out["roc_auc"] = float(roc_auc_score(y, sc))
        out["pr_auc"] = float(average_precision_score(y, sc))
    return out


def metrics_by(df: pd.DataFrame, by: str, label="label", pred="pred") -> pd.DataFrame:
    """Recall / FPR par valeur de `by` (dataset, family, session_type...)."""
    rows = []
    for key, g in df.groupby(by):
        m = binary_metrics(g[label], g[pred])
        rows.append({by: key, **{k: m[k] for k in ("n_c2", "n_benign", "recall", "fpr", "fp", "fn")}})
    return pd.DataFrame(rows)


def meets_targets(m: dict, target_recall=0.98, target_fpr=0.001) -> bool:
    return (m.get("recall", 0) >= target_recall) and (m.get("fpr", 1) < target_fpr)
