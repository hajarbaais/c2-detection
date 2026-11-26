"""Utilitaires communs aux experiences (chemins, chargement, modele MLP).
Place les imports du repo sur sys.path et offre un parseur d'arguments commun.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
for sub in ("p6_models/scripts", "p6_models/evaluation", "p4_features_dns_time/scripts"):
    sys.path.insert(0, str(REPO / sub))

from p6_common import feature_columns, load_dataset, to_matrix  # noqa: E402
from p6_08_feature_fusion import seq_summary  # noqa: E402
from metrics import metrics_by, threshold_at_fpr, wilson_ci  # noqa: E402

RANDOM_STATE = 42


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--p2", required=True, help="dossier P2 (sessions_pcap.parquet, events*.parquet)")
    ap.add_argument("--p5", default=str(REPO / "p5_features_tls_seq" / "output"),
                    help="dossier P5 (features_tls.parquet, features_seq.npz)")
    ap.add_argument("--unified", required=True,
                    help="sessions_unified.parquet de P3 (colonne ctu13_broad_label)")
    return ap


def load_all(a, with_seq=True):
    """Charge features P4+P5 (+ resume sequences) + drapeau P3, retire les C2 'convention'."""
    df = load_dataset(Path(a.p2), p5=Path(a.p5))
    if with_seq:
        sq = seq_summary(Path(a.p5))
        df = df.merge(sq, on="session_id", how="left")
        df.attrs["seq_cols"] = [c for c in sq.columns if c != "session_id"]
    flag = pd.read_parquet(a.unified, columns=["session_id", "ctu13_broad_label"])
    df = df.merge(flag, on="session_id", how="left")
    df["ctu13_broad_label"] = df["ctu13_broad_label"].fillna(False)
    df = df[~((df.label == 1) & df.ctu13_broad_label)].reset_index(drop=True)
    return df


class MLP(nn.Module):
    def __init__(self, d, hidden=(128, 64)):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X, y, tr, va, hidden=(128, 64), epochs=130, patience=12, batch=512):
    import copy
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
    tr_idx = np.flatnonzero(tr)
    torch.manual_seed(RANDOM_STATE)
    model = MLP(X.shape[1], hidden)
    pos, neg = float(y[tr].sum()), float((y[tr] == 0).sum())
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(neg / max(pos, 1), 20.0)))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(RANDOM_STATE)
    best, bad, bstate = 1e9, 0, None
    for _ in range(epochs):
        model.train()
        for i in range(0, len(tr_idx), batch):
            idx = rng.permutation(tr_idx)[:batch] if i == 0 else tr_idx[rng.integers(0, len(tr_idx), batch)]
            opt.zero_grad(); bce(model(Xt[idx]), yt[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = bce(model(Xt[va]), yt[va]).item()
        if vl < best - 1e-4:
            best, bad, bstate = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(bstate)
    with torch.no_grad():
        return model(Xt).numpy()
