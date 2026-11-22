"""P6.5 - Transformer sur les séquences P5 (features_seq.npz), suivi dans MLflow.

Auteure : Imane (P5/P6.5).  Même conventions que p5_6b_autoencoder_train.py :
un modèle par groupe (dns / web), seed 42, seuil fixé sur le BÉNIN de VALIDATION à 0,1 % de FPR,
le test n'est jamais utilisé pour choisir quoi que ce soit.

Deux modes (à choisir avec Wassima pour que LSTM et Transformer soient comparables) :
  --mode supervised  classifieur C2 / bénin, entraîné sur train (C2 + bénin), BCE avec poids de classe.
                     Score = logit (pas la probabilité : elle sature à 1,0 et rend le seuil inutilisable).
  --mode anomaly     auto-encodeur de séquences, entraîné sur le BÉNIN du train uniquement.
                     Score = erreur de reconstruction (MSE sur les pas valides).

Test des raccourcis de dataset (README racine, sprint 3) :
  --subset all       toutes les sessions
  --subset packets   web : ne garde que les sessions avec de VRAIS paquets (écarte les sessions dont les
                     pas sont uniquement des connexions « fallback » : direction = 0 et is_dns = 0).
                     Train, val, test ET seuil sont recalculés sur ce sous-ensemble.
  Si les scores s'effondrent avec `packets`, le modèle reconnaissait la capture, pas le C2.

Entrée  : <p5>/features_seq.npz  (X (N, L, F) déjà standardisé par P5, mask, label, split, session_type)
          --p2 (optionnel) : dossier avec sessions_pcap.parquet -> métriques par famille et unseen_strict.
Sorties : p6_models/models/<out-name>.pt                      (gitignoré, à partager par Drive)
          p6_models/evaluation/transformer_<mode>_<subset>_metrics.csv
          p6_models/evaluation/transformer_<mode>_<subset>_by_family.csv   (avec --p2)
          p6_models/evaluation/transformer_<mode>_<subset>_benchmark.csv   (CPU : latence, taille)

Usage :
  python -I p6_models/scripts/p6_05_transformer.py --p5 p5_features_tls_seq/output --p2 <dossier output P2> \
         --mode supervised --subset all
  python -I p6_models/scripts/p6_05_transformer.py --p5 p5_features_tls_seq/output --mode supervised --subset packets
  python -I p6_models/scripts/p6_05_transformer.py ... --mode supervised --out-name transformer   # version finale
  mlflow ui --backend-store-uri sqlite:///mlflow.db

Pour brancher dans le pipeline P7 / la fusion : load_transformer() puis score_transformer(loaded, X, mask, groups)
-> colonnes `transformer_score` (élevé = suspect) et `transformer_alert`.
"""
from __future__ import annotations

import argparse
import copy
import io
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from p6_common import (EVAL_DIR, GROUPS, MODELS_DIR, RANDOM_STATE, REPO, TARGET_FPR,  # noqa: E402
                       git_commit, setup_mlflow)
from metrics import binary_metrics, metrics_by, threshold_at_fpr  # noqa: E402

DEFAULT_P5 = REPO / "p5_features_tls_seq" / "output"


# --------------------------------------------------------------------------- données
def load_seq(p5_dir: Path, p2_dir: Path | None = None):
    """-> X (N, L, F) float32, mask (N, L) bool, meta DataFrame (1 ligne par session, même ordre)."""
    d = np.load(Path(p5_dir) / "features_seq.npz", allow_pickle=True)
    X, mask = d["X"].astype(np.float32), d["mask"].astype(bool)
    names = [str(n) for n in d["feature_names"]]
    meta = pd.DataFrame({"session_id": d["session_id"].astype(str), "label": d["label"].astype(int),
                         "split": d["split"].astype(str), "session_type": d["session_type"].astype(str)})
    meta["group"] = meta["session_type"].map({t: g for g, ts in GROUPS.items() for t in ts})

    # session « fallback » : tous ses pas sont des connexions (pas de direction de paquet, pas de DNS)
    di, dn = names.index("direction"), names.index("is_dns")
    nvalid = mask.sum(1)
    fb = (mask & (X[:, :, di] == 0) & (X[:, :, dn] == 0)).sum(1)
    meta["empty"] = nvalid == 0
    meta["fallback_only"] = (nvalid > 0) & (fb == nvalid)
    meta["n_steps"] = nvalid

    if p2_dir:
        cols = ["session_id", "family", "unseen_strict"]
        s = pd.read_parquet(Path(p2_dir) / "sessions_pcap.parquet", columns=cols)
        meta = meta.merge(s, on="session_id", how="left", validate="one_to_one")
    return X, mask, meta, names


def fix_empty(mask: np.ndarray) -> np.ndarray:
    """Une session sans aucun pas valide donnerait des NaN dans l'attention : on laisse 1 pas (X = 0)."""
    m = mask.copy()
    m[~m.any(axis=1), 0] = True
    return m


# --------------------------------------------------------------------------- modèle
class SeqTransformer(nn.Module):
    def __init__(self, n_feat: int, seq_len: int, mode: str, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 128, dropout: float = 0.1, z_dim: int = 16):
        super().__init__()
        self.dims = dict(n_feat=n_feat, seq_len=seq_len, mode=mode, d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, d_ff=d_ff, dropout=dropout, z_dim=z_dim)
        self.mode, self.seq_len, self.n_feat = mode, seq_len, n_feat
        self.inp = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        if mode == "supervised":
            self.head = nn.Linear(d_model, 1)
        else:  # anomaly : goulot d'étranglement z puis reconstruction de toute la séquence
            self.to_z = nn.Linear(d_model, z_dim)
            self.dec = nn.Sequential(nn.Linear(z_dim, 128), nn.ReLU(), nn.Linear(128, seq_len * n_feat))

    def pooled(self, x, mask):
        h = self.enc(self.inp(x) + self.pos, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)       # moyenne sur les pas valides

    def forward(self, x, mask):
        p = self.pooled(x, mask)
        if self.mode == "supervised":
            return self.head(p).squeeze(-1)                    # logit
        return self.dec(self.to_z(p)).view(-1, self.seq_len, self.n_feat)


def recon_per_session(rec, x, mask):
    """MSE par session, moyenne sur les pas valides et les features."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (((rec - x) ** 2) * m).sum((1, 2)) / (m.sum((1, 2)) * x.shape[2]).clamp(min=1.0)


@torch.no_grad()
def predict(model, X: torch.Tensor, mask: torch.Tensor, device, batch: int = 1024) -> np.ndarray:
    """Score par session : logit (supervised) ou erreur de reconstruction (anomaly). Élevé = suspect."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        x, m = X[i:i + batch].to(device), mask[i:i + batch].to(device)
        o = model(x, m)
        out.append((o if model.mode == "supervised" else recon_per_session(o, x, m)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def train_model(model, Xtr, Mtr, ytr, Xva, Mva, yva, a, device):
    """Entraînement avec arrêt anticipé sur la perte de validation. Renvoie l'historique [(ep, train, val)]."""
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    pos_w = None
    if model.mode == "supervised":
        n_pos, n_neg = float(ytr.sum()), float((ytr == 0).sum())
        pos_w = torch.tensor(min(n_neg / max(n_pos, 1.0), a.pos_weight_max), device=device)
        print(f"    pos_weight = {pos_w.item():.2f}  (train : {int(n_pos)} C2 / {int(n_neg)} bénins)")
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w) if pos_w is not None else None
    gen = torch.Generator().manual_seed(RANDOM_STATE)

    def batch_loss(x, m, y):
        o = model(x, m)
        return bce(o, y) if model.mode == "supervised" else recon_per_session(o, x, m).mean()

    @torch.no_grad()
    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        for i in range(0, len(Xva), 1024):
            x, m, y = Xva[i:i + 1024].to(device), Mva[i:i + 1024].to(device), yva[i:i + 1024].to(device)
            tot += batch_loss(x, m, y).item() * len(x)
            n += len(x)
        return tot / max(n, 1)

    hist, best, best_state, bad = [], float("inf"), None, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        tot, n = 0.0, 0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            x, m, y = Xtr[idx].to(device), Mtr[idx].to(device), ytr[idx].to(device)
            tgt = x
            if a.jitter_std > 0:                       # bruit sur log_dt (jitter), cible = séquence propre
                x = x.clone()
                x[:, :, 0] += torch.randn_like(x[:, :, 0]) * a.jitter_std * m
            o = model(x, m)
            loss = bce(o, y) if model.mode == "supervised" else recon_per_session(o, tgt, m).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        vl = val_loss()
        hist.append((ep, tot / n, vl))
        print(f"    époque {ep:3d}  train {tot / n:.4f}  val {vl:.4f}")
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= a.patience:
                print(f"    arrêt anticipé (patience {a.patience})")
                break
    model.load_state_dict(best_state)
    return hist


def benchmark_cpu(model, X: torch.Tensor, mask: torch.Tensor, threads: int) -> dict:
    """Latence CPU (ce qui compte pour la démo : < 5 s), taille du modèle, nombre de paramètres."""
    torch.set_num_threads(threads)
    m = copy.deepcopy(model).cpu().eval()
    out = {"n_params": int(sum(p.numel() for p in m.parameters())), "cpu_threads": threads}
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    out["size_mb"] = round(buf.getbuffer().nbytes / 1e6, 3)
    n = len(X)
    for bs, reps in ((1, 50), (256, 10)):
        if n < bs:
            continue
        xb, mb = X[:bs], mask[:bs]
        with torch.inference_mode():
            for _ in range(3):
                m(xb, mb)
            t0 = time.perf_counter()
            for _ in range(reps):
                m(xb, mb)
        out[f"ms_per_session_batch{bs}"] = round((time.perf_counter() - t0) / reps / bs * 1000, 4)
    return out


# --------------------------------------------------------------------------- un groupe
def run_group(group, X, mask, meta, a, device, mlflow):
    sel = meta["group"].eq(group).to_numpy()
    if group == "web" and a.subset == "packets":
        sel = sel & (~meta["fallback_only"] & ~meta["empty"]).to_numpy()
    g = meta[sel].reset_index(drop=True)
    L = min(a.max_len, X.shape[1])               # troncature : DNS a ~2 pas valides en moyenne sur 64
    Xg = torch.from_numpy(np.ascontiguousarray(X[sel][:, :L]))
    Mg = torch.from_numpy(fix_empty(mask[sel][:, :L]))
    yg = torch.from_numpy(g["label"].to_numpy().astype(np.float32))
    sp = g["split"].to_numpy()
    lab = g["label"].to_numpy()

    tr = np.flatnonzero((sp == "train") & ((lab == 0) if a.mode == "anomaly" else True))
    if a.sample_frac < 1:
        rng = np.random.default_rng(RANDOM_STATE)
        tr = rng.choice(tr, size=int(len(tr) * a.sample_frac), replace=False)
    va = np.flatnonzero((sp == "val") & ((lab == 0) if a.mode == "anomaly" else True))
    va_ben = np.flatnonzero((sp == "val") & (lab == 0))
    print(f"\n=== {group} ({a.mode}, {a.subset}) : train {len(tr):,} (C2 {int(lab[tr].sum()):,}), "
          f"val {len(va):,}, val bénin {len(va_ben):,}")
    if len(va_ben) == 0 or len(tr) == 0 or (a.mode == "supervised" and lab[tr].sum() == 0):
        print("    ! pas assez de données pour ce groupe : ignoré")
        return None, [], None, None

    model = SeqTransformer(X.shape[2], L, a.mode, a.d_model, a.heads, a.layers, a.d_ff,
                           a.dropout, a.z_dim).to(device)
    t0 = time.perf_counter()
    hist = train_model(model, Xg[tr], Mg[tr], yg[tr], Xg[va], Mg[va], yg[va], a, device)
    fit_s = time.perf_counter() - t0

    scores = predict(model, Xg, Mg, device)
    thr = threshold_at_fpr(scores[va_ben], TARGET_FPR)
    if len(va_ben) * TARGET_FPR < 5:
        print(f"    ! {len(va_ben) * TARGET_FPR:.1f} fausses alertes autorisées en val : seuil peu fiable "
              f"(voir docs/demande_P2_rebalancement.md)")

    rows, fam = [], None
    for split in ["train", "val", "test", "unseen"]:
        m = sp == split
        if not m.any():
            continue
        rows.append({"group": group, "split": split, **binary_metrics(lab[m], scores[m] > thr, scores[m])})
        if split == "test" and "family" in g:
            fam = metrics_by(g[m].assign(pred=scores[m] > thr), "family").assign(group=group)
    if "unseen_strict" in g:
        m = (sp == "unseen") & g["unseen_strict"].fillna(False).astype(bool).to_numpy()
        if m.any():
            rows.append({"group": group, "split": "unseen_strict",
                         **binary_metrics(lab[m], scores[m] > thr, scores[m])})

    bench = benchmark_cpu(model, Xg[sp == "test"] if (sp == "test").any() else Xg,
                          Mg[sp == "test"] if (sp == "test").any() else Mg, a.threads)
    bench["group"] = group
    print(f"    {len(hist)} époques en {fit_s:.1f} s ; seuil = {thr:.4f} ; CPU : "
          f"{bench.get('ms_per_session_batch1')} ms/session (lot 1), {bench['size_mb']} Mo, {bench['n_params']:,} paramètres")

    if mlflow:
        mlflow.log_params({f"{group}_n_train": len(tr), f"{group}_n_val_benign": len(va_ben),
                           f"{group}_epochs_run": len(hist)})
        mlflow.log_metric(f"{group}/threshold", float(thr))
        mlflow.log_metric(f"{group}/fit_seconds", fit_s)
        for k in ("size_mb", "n_params", "ms_per_session_batch1", "ms_per_session_batch256"):
            if k in bench:
                mlflow.log_metric(f"cpu/{group}/{k}", float(bench[k]))
        for ep, tl, vl in hist:
            mlflow.log_metric(f"train/{group}/loss", tl, step=ep)
            mlflow.log_metric(f"val/{group}/loss", vl, step=ep)
        for r in rows:
            for k in ("recall", "fpr", "precision", "f1", "roc_auc", "pr_auc"):
                v = r.get(k)
                if v is not None and not np.isnan(v):
                    mlflow.log_metric(f"{r['split']}/{group}/{k}", float(v))

    bundle = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "dims": model.dims,
              "threshold": float(thr), "history": hist}
    return bundle, rows, fam, bench


# --------------------------------------------------------------------------- rechargement (pipeline P7 / fusion)
def load_transformer(path, device="cpu") -> dict:
    """Recharge le .pt -> {groupe: {model, threshold, mode}} (une seule fois au démarrage)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    out = {}
    for group, b in ckpt["groups"].items():
        model = SeqTransformer(**b["dims"])
        model.load_state_dict(b["state_dict"])
        model.eval()
        out[group] = {"model": model.to(device), "threshold": b["threshold"], "mode": ckpt["mode"]}
    return out


def score_transformer(loaded: dict, X: np.ndarray, mask: np.ndarray, groups, device="cpu") -> pd.DataFrame:
    """X (N, L, F) et mask (N, L) au format P5 (build_sequences + scale_sequences) ; `groups` = 'dns'/'web' par session.

    Renvoie transformer_score (élevé = suspect) et transformer_alert, dans l'ordre des sessions.
    """
    groups = np.asarray(groups)
    out = pd.DataFrame({"transformer_score": np.nan, "transformer_alert": False}, index=range(len(X)))
    for group, b in loaded.items():
        idx = np.flatnonzero(groups == group)
        if idx.size == 0:
            continue
        L = b["model"].seq_len                    # le modèle a pu être entraîné sur des séquences tronquées
        xs = torch.from_numpy(np.ascontiguousarray(X[idx][:, :L], dtype=np.float32))
        ms = torch.from_numpy(fix_empty(np.asarray(mask[idx][:, :L], dtype=bool)))
        s = predict(b["model"], xs, ms, device)
        out.loc[idx, "transformer_score"] = s
        out.loc[idx, "transformer_alert"] = s > b["threshold"]
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p5", default=str(DEFAULT_P5), help="dossier contenant features_seq.npz")
    ap.add_argument("--p2", help="dossier P2 (sessions_pcap.parquet) : métriques par famille + unseen_strict")
    ap.add_argument("--mode", choices=["supervised", "anomaly"], required=True)
    ap.add_argument("--subset", choices=["all", "packets"], default="all")
    ap.add_argument("--groups", default="dns,web")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--z-dim", type=int, default=16, help="anomaly : taille du goulot")
    ap.add_argument("--pos-weight-max", type=float, default=20.0, help="supervised : plafond du poids de la classe C2")
    ap.add_argument("--max-len", type=int, default=64,
                    help="tronque les séquences aux N premiers pas (ex. 16 pour dns : beaucoup plus rapide)")
    ap.add_argument("--jitter-std", type=float, default=0.0, help="bruit gaussien sur log_dt pendant l'entraînement")
    ap.add_argument("--sample-frac", type=float, default=1.0, help="fraction du train (test rapide)")
    ap.add_argument("--threads", type=int, default=1, help="threads CPU pour le benchmark de latence")
    ap.add_argument("--out-name", help="nom du .pt (défaut : transformer_<mode>_<subset>) ; 'transformer' pour la version finale")
    ap.add_argument("--run-name", help="nom du run MLflow")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device d'entraînement : {device}  (le benchmark de latence est toujours sur CPU)")

    X, mask, meta, names = load_seq(Path(a.p5), Path(a.p2) if a.p2 else None)
    print(f"features_seq : X{X.shape}, features = {names}")

    tag = f"{a.mode}_{a.subset}"
    out_name = a.out_name or f"transformer_{tag}"
    mlflow = None if a.no_mlflow else setup_mlflow("c2-transformer")
    run = mlflow.start_run(run_name=a.run_name or tag) if mlflow else nullcontext()
    bundles, rows, fams, benches = {}, [], [], []
    with run:
        if mlflow:
            mlflow.set_tags({"lot": "P6", "task": "P6.5", "author": "imane", "git_commit": git_commit(),
                             "features": "p5_features_tls_seq (features_seq.npz)", "mode": a.mode,
                             "subset": a.subset})
            mlflow.log_params({"mode": a.mode, "subset": a.subset, "d_model": a.d_model, "heads": a.heads,
                               "layers": a.layers, "d_ff": a.d_ff, "dropout": a.dropout, "lr": a.lr,
                               "batch": a.batch, "epochs_max": a.epochs, "patience": a.patience,
                               "jitter_std": a.jitter_std, "target_fpr": TARGET_FPR,
                               "threshold_split": "val_benign", "seed": RANDOM_STATE})
        for group in [x.strip() for x in a.groups.split(",") if x.strip()]:
            b, r, fam, bench = run_group(group, X, mask, meta, a, device, mlflow)
            if b is None:
                continue
            bundles[group] = b
            rows += r
            if fam is not None:
                fams.append(fam)
            benches.append(bench)

        if not bundles:
            sys.exit("[!] aucun groupe entraîné")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        pt = MODELS_DIR / f"{out_name}.pt"
        torch.save({"groups": bundles, "mode": a.mode, "subset": a.subset, "seed": RANDOM_STATE,
                    "target_fpr": TARGET_FPR, "features": names}, pt)
        res = pd.DataFrame(rows)
        res.to_csv(EVAL_DIR / f"transformer_{tag}_metrics.csv", index=False, float_format="%.6f")
        pd.DataFrame(benches).to_csv(EVAL_DIR / f"transformer_{tag}_benchmark.csv", index=False)
        if fams:
            pd.concat(fams).to_csv(EVAL_DIR / f"transformer_{tag}_by_family.csv", index=False, float_format="%.6f")
        if mlflow:
            mlflow.log_artifact(str(pt), artifact_path="models")
            mlflow.log_artifact(str(EVAL_DIR / f"transformer_{tag}_metrics.csv"))

    print(f"\nModèle : {pt}")
    view = res[res.split.isin(["val", "test", "unseen", "unseen_strict"])][
        ["group", "split", "n_c2", "n_benign", "recall", "fpr", "roc_auc"]]
    print(view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nObjectifs : recall >= 0,98 et FPR < 0,001 (à mesurer sur le test, jamais à annoncer avant).")
    print("Raccourcis : relancer avec --subset packets et comparer.")


if __name__ == "__main__":
    main()
