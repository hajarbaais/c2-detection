"""P5.6a - Prototype d'autoencodeur (PyTorch) entraîné sur un ÉCHANTILLON du trafic bénin.

But du prototype : vérifier que l'architecture, le prétraitement, le seuil et l'évaluation
fonctionnent de bout en bout, avec les mêmes règles que Wassima (P6.1 / P6.2) pour que les
résultats soient comparables à l'Isolation Forest.

Principe
    - un modèle par groupe (dns / web), comme P6 ;
    - entraînement UNIQUEMENT sur le bénin du train (sous-échantillonné par --sample-frac) ;
    - score = erreur de reconstruction (MSE par session) ; score élevé = suspect ;
    - seuil fixé sur le bénin de VALIDATION pour 0,1 % de FPR (le test n'est jamais utilisé
      pour choisir quoi que ce soit) ;
    - scaler / médianes / log1p appris sur le bénin du train seulement.

Entrées : <P2>/sessions_pcap.parquet, p4_features_dns_time/output/features_dns.parquet,
          (option --p5) <p5_output>/features_tls.parquet
Sorties : p6_models/output/autoencoder_proto_<groupe>.pt   (dossier gitignoré)
          p6_models/output/autoencoder_proto_metrics.csv

Usage : python -I p6_models/scripts/p5_6a_autoencoder_proto.py --p2 <dossier output P2>
            [--p5 <dossier p5_output>] [--sample-frac 0.2] [--epochs 30]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from p6_common import (GROUPS, RANDOM_STATE, REPO, SESSION_NUMERIC, TARGET_FPR,  # noqa: E402
                       feature_columns, load_dataset, p2_dir)
from metrics import binary_metrics, metrics_by, threshold_at_fpr  # noqa: E402

OUT_DIR = REPO / "p6_models" / "output"
# 14 features tabulaires de P5 (Imane), utilisées pour le groupe web
P5_COLS = ["log_orig_bytes", "log_resp_bytes", "up_down_ratio", "avg_pkt_size", "log_duration",
           "log_n_conn", "log_n_conn_no_payload", "conn_rate", "has_ja3", "ja3_rarity", "ja4_rarity",
           "tls_version", "cert_self_signed", "log_domain_reputation"]
# sessions_pcap contient déjà `tls_version` et `cert_self_signed` (valeurs brutes) : on préfixe les
# colonnes P5 par `p5_` à la jointure pour ne rien écraser.
P5_FEATS = ["p5_" + c for c in P5_COLS]
# ROC-AUC test de l'Isolation Forest (iforest_metrics.csv) : le chiffre à battre
IFOREST_AUC = {"dns": 0.804, "web": 0.924}


# --------------------------------------------------------------------------- prétraitement
class Prep:
    """log1p (colonnes à queue lourde) -> indicateurs de NaN -> médiane -> standardisation.

    Tout est appris sur le bénin du train. to_dict() donne une version sans classe perso, pour
    recharger le modèle ailleurs (pipeline P7) sans dépendre de ce fichier.
    """

    def fit(self, X: pd.DataFrame) -> "Prep":
        X = X.astype("float64")
        self.cols = list(X.columns)
        # log1p seulement si la colonne est positive et très asymétrique (et pas déjà un log_*)
        self.log_cols = [c for c in self.cols
                         if not c.startswith("log_") and X[c].min() >= 0 and X[c].skew() > 2]
        X = self._log(X)
        rate = X.isna().mean()
        self.miss_cols = [c for c in self.cols if 0.001 < rate[c] < 0.999]
        self.median = X.median().fillna(0.0)
        Xf = X.fillna(self.median)
        self.mean = Xf.mean()
        self.std = Xf.std().fillna(1.0).replace(0.0, 1.0)
        return self

    def _log(self, X):
        X = X.copy()
        for c in self.log_cols:
            X[c] = np.log1p(X[c].clip(lower=0))
        return X

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._log(X[self.cols].astype("float64"))
        ind = X[self.miss_cols].isna().astype("float64").to_numpy()
        Z = ((X.fillna(self.median) - self.mean) / self.std).clip(-10, 10).to_numpy()
        return np.hstack([Z, ind]).astype(np.float32)

    @property
    def n_out(self) -> int:
        return len(self.cols) + len(self.miss_cols)

    def to_dict(self) -> dict:
        return {"cols": self.cols, "log_cols": self.log_cols, "miss_cols": self.miss_cols,
                "median": self.median.to_dict(), "mean": self.mean.to_dict(), "std": self.std.to_dict()}


# --------------------------------------------------------------------------- modèle
class AutoEncoder(nn.Module):
    """MLP symétrique : d -> 64 -> 32 -> z -> 32 -> 64 -> d (z = goulot d'étranglement)."""

    def __init__(self, d: int, h1: int = 64, h2: int = 32, z: int = 8):
        super().__init__()
        z = max(2, min(z, d // 3))
        self.dims = dict(d=d, h1=h1, h2=h2, z=z)
        self.enc = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, z))
        self.dec = nn.Sequential(nn.Linear(z, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, d))

    def forward(self, x):
        return self.dec(self.enc(x))


@torch.no_grad()
def recon_error(model: nn.Module, X: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Score d'anomalie = erreur quadratique moyenne de reconstruction, une valeur par session."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch])
        out.append(((model(xb) - xb) ** 2).mean(dim=1).numpy())
    return np.concatenate(out) if out else np.array([])


def train(model, Xtr, Xva, epochs, lr, batch, patience):
    """Adam + arrêt anticipé sur la perte de reconstruction du bénin de validation."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ds = torch.from_numpy(Xtr)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(ds))
        tot = 0.0
        for i in range(0, len(ds), batch):
            xb = ds[perm[i:i + batch]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        tr_loss, va_loss = tot / len(ds), float(recon_error(model, Xva).mean())
        hist.append((ep, tr_loss, va_loss))
        if va_loss < best - 1e-6:
            best, wait = va_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return hist


# --------------------------------------------------------------------------- un groupe
def run_group(df, group, p5: bool, a):
    g = df[df["group"] == group].reset_index(drop=True)
    cols = feature_columns(df, group)
    if group == "web" and p5:
        cols = [c for c in cols if c not in SESSION_NUMERIC] + P5_FEATS
    tr = g[(g["split"] == "train") & (g["label"] == 0)]
    if a.sample_frac < 1:
        tr = tr.sample(frac=a.sample_frac, random_state=RANDOM_STATE)
    va = g[(g["split"] == "val") & (g["label"] == 0)]
    print(f"\n=== groupe {group} : {len(cols)} colonnes, train bénin {len(tr):,}, val bénin {len(va):,}")

    prep = Prep().fit(tr[cols])
    Xtr, Xva = prep.transform(tr[cols]), prep.transform(va[cols])
    model = AutoEncoder(prep.n_out)
    print(f"    entrée {prep.n_out} (dont {len(prep.miss_cols)} indicateurs NaN) ; dims {model.dims} ; "
          f"{sum(p.numel() for p in model.parameters()):,} paramètres ; log1p sur {len(prep.log_cols)} colonnes")
    t0 = time.perf_counter()
    hist = train(model, Xtr, Xva, a.epochs, a.lr, a.batch, a.patience)
    print(f"    {len(hist)} époques en {time.perf_counter() - t0:.1f} s ; perte train {hist[0][1]:.4f} -> "
          f"{hist[-1][1]:.4f} ; perte val {hist[0][2]:.4f} -> {hist[-1][2]:.4f}")

    thr = threshold_at_fpr(recon_error(model, Xva), TARGET_FPR)
    if len(va) * TARGET_FPR < 5:
        print(f"    ! seulement {len(va) * TARGET_FPR:.1f} faux positifs autorisés en val : seuil peu fiable")

    rows, test_pred = [], None
    for split in ["val", "test", "unseen"]:
        s = g[g["split"] == split]
        sc = recon_error(model, prep.transform(s[cols]))
        pred = sc > thr
        rows.append({"group": group, "split": split, **binary_metrics(s["label"], pred, sc)})
        if split == "test":
            test_pred = s.assign(pred=pred)
    strict = g[(g["split"] == "unseen") & (g["unseen_strict"] == True)]  # noqa: E712
    if len(strict):
        r = binary_metrics(strict["label"], recon_error(model, prep.transform(strict[cols])) > thr)
        rows.append({"group": group, "split": "unseen_strict", **r})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "dims": model.dims, "prep": prep.to_dict(),
                "threshold": float(thr), "group": group, "seed": RANDOM_STATE,
                "history": hist}, OUT_DIR / f"autoencoder_proto_{group}.pt")
    return rows, test_pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2", help="dossier output de P2 (sessions_pcap.parquet)")
    ap.add_argument("--p5", help="dossier p5_output (features_tls.parquet) : ajoute les features TLS au groupe web")
    ap.add_argument("--sample-frac", type=float, default=0.2, help="part du bénin du train utilisée (prototype)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--groups", nargs="+", default=list(GROUPS))
    a = ap.parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    df = load_dataset(p2_dir(a.p2)).reset_index(drop=True)
    if a.p5:
        t = pd.read_parquet(Path(a.p5) / "features_tls.parquet")[["session_id"] + P5_COLS]
        t = t.rename(columns={c: "p5_" + c for c in P5_COLS})
        df = df.merge(t, on="session_id", how="left", validate="one_to_one")

    all_rows = []
    for group in a.groups:
        rows, test_pred = run_group(df, group, bool(a.p5), a)
        all_rows += rows
        fam = metrics_by(test_pred, "family")
        print("    recall par famille C2 (test) :",
              {r.family: round(r.recall, 3) for r in fam.itertuples() if r.n_c2 > 0})
        print("    FPR par famille bénigne (test) :",
              {r.family: round(r.fpr, 4) for r in fam.itertuples() if r.n_benign > 0})

    res = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / "autoencoder_proto_metrics.csv", index=False, float_format="%.6f")
    print("\n=== résumé (seuil à 0,1 % de FPR sur le bénin de validation)")
    view = res[["group", "split", "n_c2", "n_benign", "recall", "fpr", "roc_auc"]].copy()
    view["iforest_auc"] = [IFOREST_AUC.get(g) if s == "test" else None for g, s in zip(view.group, view.split)]
    print(view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nObjectifs du prof : recall >= 0,98 et FPR < 0,001 (à mesurer sur le test, jamais à annoncer avant).")


if __name__ == "__main__":
    main()
