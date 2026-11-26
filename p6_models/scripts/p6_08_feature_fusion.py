"""P6.8 (experimental) - Fusion de FEATURES : UN SEUL modele supervise nourri avec TOUTES les features.

Au lieu de fusionner les SCORES de plusieurs modeles (P6.6, fusion tardive), on reunit toutes les
features dans un seul tableau et on entraine un seul classifieur (fusion precoce) :
  - features statiques P4 (entropie, forme, IAT, periodicite, fenetres)   -> Hajar
  - features statiques P5 (14 features TLS : rarete JA3/JA4, reputation...) -> Imane
  - resume des sequences de paquets (moyenne / ecart-type / max de chaque canal, sur les pas valides)

Un seul modele voit les INTERACTIONS entre tous les indices -> recall superieur (cf. dridex 0->100 %).
Sans scikit-learn (classifieur PyTorch), donc executable malgre le blocage sklearn.

Seuil : bénin de VALIDATION a 0,1 % de FPR (le test n'est jamais utilise pour choisir). Seed 42.
On reporte aussi le recall a plusieurs FPR (0,1 / 0,5 / 1 %) pour montrer le compromis.

Entrees : features_dns.parquet (P4) + <p5>/features_tls.parquet (P5) + <p5>/features_seq.npz (sequences)
          --p2 : sessions_pcap.parquet (labels, familles, split)
Sorties : p6_models/evaluation/feature_fusion_metrics.csv  +  feature_fusion_by_family.csv
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from p6_common import EVAL_DIR, RANDOM_STATE, TARGET_FPR, feature_columns, load_dataset, to_matrix  # noqa: E402
from metrics import binary_metrics, metrics_by, threshold_at_fpr  # noqa: E402

DEFAULT_P5 = Path(__file__).resolve().parents[1].parent / "p5_features_tls_seq" / "output"


def seq_summary(p5_dir: Path) -> pd.DataFrame:
    """Resume chaque session : moyenne / ecart-type / max de chaque canal de sequence sur les pas valides."""
    d = np.load(p5_dir / "features_seq.npz", allow_pickle=True)
    X, mask = d["X"].astype(np.float64), d["mask"].astype(bool)
    names = [str(n) for n in d["feature_names"]]
    sid = d["session_id"].astype(str)
    m = mask[:, :, None]
    nvalid = mask.sum(1).clip(min=1)[:, None]
    mean = (X * m).sum(1) / nvalid
    var = ((X - mean[:, None, :]) ** 2 * m).sum(1) / nvalid
    std = np.sqrt(var)
    Xmax = np.where(mask[:, :, None], X, -np.inf).max(1)
    Xmax[~np.isfinite(Xmax)] = 0.0
    out = {"session_id": sid}
    for j, n in enumerate(names):
        out[f"seq_{n}_mean"] = mean[:, j]
        out[f"seq_{n}_std"] = std[:, j]
        out[f"seq_{n}_max"] = Xmax[:, j]
    out["seq_n_steps"] = mask.sum(1)
    return pd.DataFrame(out)


class MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_group(group: str, df: pd.DataFrame, seq_cols: list[str], a) -> tuple[dict, pd.DataFrame]:
    static_cols = feature_columns(df, group, with_p5=True)
    cols = static_cols + seq_cols
    g = df[df.group == group].copy()
    Xs = to_matrix(g, static_cols).astype("float32")           # log1p + NaN -> -1 (comme le pipeline)
    Xq = g[seq_cols].to_numpy("float32")
    Xq = np.nan_to_num(Xq, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.concatenate([Xs, Xq], axis=1)
    y = g.label.to_numpy().astype("float32")
    sp = g.split.to_numpy()
    tr, va, te = sp == "train", sp == "val", sp == "test"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
    tr_idx = np.flatnonzero(tr)

    torch.manual_seed(RANDOM_STATE)
    model = MLP(X.shape[1])
    pos, neg = float(y[tr].sum()), float((y[tr] == 0).sum())
    posw = torch.tensor(min(neg / max(pos, 1), 20.0))
    bce = nn.BCEWithLogitsLoss(pos_weight=posw)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"\n=== {group} : {len(static_cols)} features statiques + {len(seq_cols)} resume-sequence = {X.shape[1]} ; "
          f"train {tr.sum():,} (C2 {int(pos)}), test {te.sum():,}")

    rng = np.random.default_rng(RANDOM_STATE)
    best, bad, best_state = 1e9, 0, None
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            opt.zero_grad()
            bce(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = bce(model(Xt[va]), yt[va]).item()
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        scores = model(Xt).numpy()

    vb = va & (y == 0)
    rows = []
    for target in (0.001, 0.005, 0.01):
        thr = threshold_at_fpr(scores[vb], target)
        m = binary_metrics(y[te], scores[te] > thr, scores[te])
        rows.append({"group": group, "target_fpr": target, "recall": m["recall"], "fpr": m["fpr"],
                     "roc_auc": m["roc_auc"], "precision": m["precision"]})
        print(f"    seuil @ FPR {target*100:>4.1f}% -> recall test {m['recall']*100:5.1f}%  "
              f"FPR reelle {m['fpr']*100:.3f}%  AUC {m['roc_auc']:.3f}")

    # famille au seuil principal (0,1 % FPR)
    thr0 = threshold_at_fpr(scores[vb], TARGET_FPR)
    fam = g[te].assign(label=y[te], pred=scores[te] > thr0)
    ff = metrics_by(fam, "family")
    ff = ff[ff.n_c2 > 0].assign(group=group).sort_values("recall", ascending=False)
    return rows, ff


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2", required=True)
    ap.add_argument("--p5", default=str(DEFAULT_P5))
    ap.add_argument("--groups", default="dns,web")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args(argv)

    print("[1/3] chargement features P4 + P5 ...")
    df = load_dataset(Path(a.p2), p5=Path(a.p5))
    print("[2/3] resume des sequences de paquets ...")
    sq = seq_summary(Path(a.p5))
    seq_cols = [c for c in sq.columns if c != "session_id"]
    df = df.merge(sq, on="session_id", how="left")

    print("[3/3] entrainement du modele de fusion de features par groupe ...")
    all_rows, fams = [], []
    for grp in [x.strip() for x in a.groups.split(",") if x.strip()]:
        rows, ff = run_group(grp, df, seq_cols, a)
        all_rows += rows
        fams.append(ff)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    met = pd.DataFrame(all_rows)
    met.to_csv(EVAL_DIR / "feature_fusion_metrics.csv", index=False, float_format="%.6f")
    fam = pd.concat(fams, ignore_index=True)
    fam.to_csv(EVAL_DIR / "feature_fusion_by_family.csv", index=False, float_format="%.6f")
    print("\n=== RECALL PAR FAMILLE (seuil @ 0,1 % FPR) ===")
    print(fam[["group", "family", "n_c2", "recall", "fpr"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n-> {EVAL_DIR / 'feature_fusion_metrics.csv'}\n-> {EVAL_DIR / 'feature_fusion_by_family.csv'}")


if __name__ == "__main__":
    main()
