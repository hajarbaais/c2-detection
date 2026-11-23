"""P7.2 — Local explanations of scores for alerted sessions."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


BACKGROUND_SIZE = 16
PERMUTATIONS = 3


def benign_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only the benign training split as a candidate reference."""
    if not {"split", "label"} <= set(frame.columns):
        return frame.iloc[:0]
    return frame.loc[frame["split"].eq("train") & frame["label"].eq(0)]


def explain_top_features(score, rows: pd.DataFrame, background: pd.DataFrame,
                         columns: list[str]) -> list[list[dict]]:
    """Explain a scalar score in its own units, using the original feature columns."""
    import shap

    def predict(values):
        return score(pd.DataFrame(values, columns=columns))

    reference = background[columns].to_numpy(dtype=float, na_value=np.nan)
    values = rows[columns].to_numpy(dtype=float, na_value=np.nan)
    masker = shap.maskers.Independent(reference, max_samples=BACKGROUND_SIZE)
    explainer = shap.PermutationExplainer(predict, masker, seed=0)
    explanation = explainer(values, max_evals=PERMUTATIONS * (2 * len(columns) + 1), silent=True)

    results = []
    for inputs, impacts in zip(values, explanation.values):
        order = np.argsort(-np.abs(impacts), kind="stable")
        results.append([
            {"name": columns[i], "value": float(inputs[i]) if np.isfinite(inputs[i]) else None,
             "impact": float(impacts[i])}
            for i in order if np.isfinite(impacts[i]) and impacts[i] != 0
        ][:3])
    return results


def explain_alerts(features: pd.DataFrame, verdicts: pd.DataFrame, reference: pd.DataFrame,
                   scorers) -> dict:
    """Explain the first triggering scorer in priority order, separately per group."""
    alerted = verdicts["alert"].eq(1).to_numpy()
    results = {}
    for group, rows in features.loc[alerted].groupby("group"):
        background = reference.loc[reference["group"].eq(group)]
        if background.empty:
            warnings.warn(f"SHAP unavailable for {group}: no benign training reference", stacklevel=2)
            continue
        background = background.sample(min(len(background), BACKGROUND_SIZE), random_state=0)
        pending = rows
        for flag, columns, score in scorers(group):
            if flag not in verdicts or not set(columns) <= set(background.columns):
                continue
            selected = pending["session_id"].isin(verdicts.loc[verdicts[flag], "session_id"])
            if not selected.any():
                continue
            batch = pending.loc[selected]
            explanations = explain_top_features(score, batch, background, columns)
            results.update(zip(batch["session_id"].astype(str), explanations))
            pending = pending.loc[~selected]
            if pending.empty:
                break
        if not pending.empty:
            warnings.warn(f"SHAP unavailable for {len(pending)} {group} alert(s): "
                          "no triggering scorer with matching reference features", stacklevel=2)
    return results
