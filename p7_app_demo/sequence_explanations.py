"""Scores and grouped SHAP for trained P5 sequence models."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class SequenceExplanations:
    def __init__(self, p5_dir, lstm_path, transformer_path):
        from p6_03_lstm import load_lstm, score_lstm
        from p6_05_transformer import load_seq, load_transformer, score_transformer

        self.values, self.mask, self.meta, self.names = load_seq(p5_dir)
        self.index = pd.Index(self.meta["session_id"])
        if not self.index.is_unique:
            raise ValueError("Duplicate session IDs in features_seq.npz")
        self.models = {}
        self.references = {}
        for name, path, load, score in (("lstm", lstm_path, load_lstm, score_lstm),
                                        ("transformer", transformer_path, load_transformer, score_transformer)):
            if path is None:
                continue
            bundles = load(path)
            for group, bundle in bundles.items():
                if bundle["features"] is not None and list(bundle["features"]) != self.names:
                    raise ValueError(f"{name}/{group}: checkpoint feature order differs from P5")
                if bundle["model"].n_feat != len(self.names):
                    raise ValueError(f"{name}/{group}: checkpoint feature count differs from P5")
                length = bundle["model"].seq_len
                if length > self.values.shape[1]:
                    raise ValueError(f"{name}/{group}: P5 sequences shorter than checkpoint input")
                eligible = self.meta["group"].eq(group)
                if group == "web" and bundle["subset"] == "packets":
                    eligible &= ~self.meta["fallback_only"] & ~self.meta["empty"]
                self.models[name, group] = (bundle, score, eligible.to_numpy())

    def add_scores(self, out, features):
        positions = self.index.get_indexer(features["session_id"].astype(str))
        present = np.flatnonzero(positions >= 0)
        for (name, group), (bundle, score, eligible) in self.models.items():
            if f"{name}_score" not in out:
                out[f"{name}_score"] = np.nan
                out[f"{name}_alert"] = False
            rows = present[eligible[positions[present]] & features["group"].eq(group).to_numpy()[present]]
            if not rows.size:
                continue
            source = positions[rows]
            result = score({group: bundle}, self.values[source], self.mask[source], [group] * len(rows))
            for col in result:
                out.iloc[rows, out.columns.get_loc(col)] = result[col].to_numpy()

    def explain(self, verdicts):
        explanations = {}
        for (name, group), (bundle, score, eligible) in self.models.items():
            column = f"{name}_alert"
            if column not in verdicts:
                continue
            alerted = verdicts.loc[verdicts["alert"].eq(1) & verdicts[column], "session_id"].astype(str)
            length = bundle["model"].seq_len
            for session_id in alerted:
                source = self.index.get_loc(session_id)
                if session_id in explanations or not eligible[source]:
                    continue
                mask = self.mask[source, :length]
                if not mask.any():
                    continue
                key = name, group
                if key not in self.references:
                    train = eligible & self.meta["split"].eq("train") & self.meta["label"].eq(0)
                    train_mask = self.mask[train, :length]
                    if not train_mask.any():
                        warnings.warn(f"SHAP unavailable for {name}/{group}: no valid benign training steps",
                                      stacklevel=2)
                        self.references[key] = None
                    else:
                        self.references[key] = np.median(self.values[train, :length][train_mask], axis=0)
                reference = self.references[key]
                if reference is not None:
                    sequence = self.values[source, :length]

                    def predict(batch, masks):
                        return score({group: bundle}, batch, masks, [group] * len(batch))[f"{name}_score"].to_numpy()

                    explanations[session_id] = explain_sequence(predict, sequence, mask, reference, self.names)
        return explanations


def explain_sequence(predict, sequence, mask, reference, names):
    """One SHAP player per channel across all valid steps; the padding mask stays fixed."""
    import shap

    def score_coalitions(gates):
        replace = ~gates[:, None, :].astype(bool) & mask[None, :, None]
        batch = np.where(replace, reference, sequence)
        masks = np.broadcast_to(mask, batch.shape[:2]).copy()
        return predict(batch, masks)

    baseline = np.zeros((1, len(names)))
    explainer = shap.ExactExplainer(score_coalitions, baseline)
    impacts = explainer(np.ones_like(baseline), silent=True).values[0]
    changed = (sequence[mask] != reference).any(axis=0)
    order = np.argsort(-np.abs(impacts), kind="stable")
    return [{"name": f"seq_{names[i]}", "value": float(sequence[mask, i].mean()), "impact": float(impacts[i])}
            for i in order if changed[i] and np.isfinite(impacts[i]) and impacts[i] != 0][:3]
