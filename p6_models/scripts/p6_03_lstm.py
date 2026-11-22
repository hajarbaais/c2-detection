"""P6.3 - LSTM sur les séquences P5 (features_seq.npz) + P6.4 entraînement robuste au jitter, suivi dans MLflow.

Auteure : Wassima (sprint 3).  Conçu pour être COMPARABLE au Transformer d'Imane (p6_05_transformer.py) :
  - mêmes données et même sélection (groupes dns / web, --subset packets, --max-len), seed 42 ;
  - même seuil : bénin de VALIDATION à 0,1 % de FPR ; le test n'est jamais utilisé pour choisir ;
  - mêmes deux modes : --mode supervised (logit, BCE avec poids de classe) / --mode anomaly (bénin seul,
    erreur de reconstruction) ;
  - en mode anomaly, MÊME goulot z = 16 et MÊME décodeur MLP que le Transformer : seule la partie
    « encodeur » change (LSTM au lieu de self-attention) -> la comparaison mesure l'encodeur, pas le reste ;
  - mêmes fichiers de sortie (metrics / by_family / benchmark CPU) et même interface pour la fusion.

Encodeur : LSTM 2 couches (hidden 64), séquences « packées » (les pas de padding ne sont jamais lus),
représentation de la session = état caché du DERNIER pas valide.

P6.4 (robustesse au jitter) :
  --jitter-train J   pendant l'entraînement, chaque session reçoit un jitter réaliste de niveau ~ U(0, J)
                     sur ses intervalles (dt x U(1-j, 1+j) en secondes, voir p6_04_jitter.py).
                     anomaly : entrée bruitée, cible = séquence propre (même convention que --jitter-std d'Imane).
                     supervised : entrée bruitée, label inchangé (bénin ET C2 bruités : pas de raccourci « bruité = C2 »).
  --jitter-eval      niveaux évalués après l'entraînement, à seuil fixe, sur les C2 du test / unseen (défaut 0,0.1,0.25,0.5).

Sorties : p6_models/models/<out-name>.pt                                  (gitignoré, à partager par Drive)
          p6_models/evaluation/lstm_<mode>_<subset>[_jitNN]_metrics.csv
          p6_models/evaluation/lstm_<tag>_benchmark.csv / _by_family.csv (--p2) / _jitter.csv / _jitter_by_family.csv
          (un essai --groups dns ne remplace que les lignes dns du CSV : le web reste)

Usage (mêmes essais que le Transformer, voir p6_models/README_lstm.md) :
  python -I p6_models/scripts/p6_03_lstm.py --p5 p5_features_tls_seq/output --p2 <P2> \
         --mode anomaly --groups web --subset packets --out-name lstm_web_anomaly_packets
  python -I p6_models/scripts/p6_03_lstm.py --p5 p5_features_tls_seq/output --p2 <P2> \
         --mode anomaly --groups web --subset packets --jitter-train 0.5 --out-name lstm_web_anomaly_packets_jit50
  mlflow ui --backend-store-uri sqlite:///mlflow.db

Pour la fusion / le pipeline P7 : load_lstm() puis score_lstm(loaded, X, mask, groups)
-> colonnes `lstm_score` (élevé = suspect) et `lstm_alert`.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from p6_common import EVAL_DIR, MODELS_DIR, RANDOM_STATE, REPO, TARGET_FPR, git_commit, setup_mlflow  # noqa: E402
from metrics import binary_metrics, metrics_by, threshold_at_fpr  # noqa: E402
from p6_04_jitter import (DEFAULT_LEVELS, JITTER_KINDS, jitter_log_dt, load_dt_scaler,  # noqa: E402
                          merge_csv, robustness, select_group)
from p6_05_transformer import benchmark_cpu, fix_empty, load_seq, predict, recon_per_session  # noqa: E402

DEFAULT_P5 = REPO / "p5_features_tls_seq" / "output"


# --------------------------------------------------------------------------- modèle
def check_prefix_mask(mask) -> bool:
    """Le LSTM « packé » suppose que les pas valides sont les premiers (rangs 0..n-1), ce que fait P5."""
    m = torch.as_tensor(mask).bool()
    n = m.sum(1, keepdim=True)
    return bool((m == (torch.arange(m.shape[1]).unsqueeze(0) < n)).all())


class SeqLSTM(nn.Module):
    """Même interface que SeqTransformer (mode, seq_len, n_feat, dims, forward(x, mask)) -> predict() d'Imane marche tel quel."""

    def __init__(self, n_feat: int, seq_len: int, mode: str, hidden: int = 64, n_layers: int = 2,
                 dropout: float = 0.1, z_dim: int = 16, bidirectional: bool = False):
        super().__init__()
        self.dims = dict(n_feat=n_feat, seq_len=seq_len, mode=mode, hidden=hidden, n_layers=n_layers,
                         dropout=dropout, z_dim=z_dim, bidirectional=bidirectional)
        self.mode, self.seq_len, self.n_feat = mode, seq_len, n_feat
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0, bidirectional=bidirectional)
        out = hidden * (2 if bidirectional else 1)
        if mode == "supervised":
            self.head = nn.Linear(out, 1)
        else:  # anomaly : même goulot + même décodeur que le Transformer
            self.to_z = nn.Linear(out, z_dim)
            self.dec = nn.Sequential(nn.Linear(z_dim, 128), nn.ReLU(), nn.Linear(128, seq_len * n_feat))

    def encode(self, x, mask):
        lengths = mask.sum(1).clamp(min=1).to("cpu", torch.int64)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)                 # h : (couches * directions, B, hidden), ordre d'origine
        return torch.cat([h[-2], h[-1]], 1) if self.bidirectional else h[-1]

    def forward(self, x, mask):
        p = self.encode(x, mask)
        if self.mode == "supervised":
            return self.head(p).squeeze(-1)           # logit
        return self.dec(self.to_z(p)).view(-1, self.seq_len, self.n_feat)


# --------------------------------------------------------------------------- entraînement
def train_model(model, Xtr, Mtr, ytr, Xva, Mva, yva, a, device, dt_scaler=None):
    """Arrêt anticipé sur la perte de validation (propre). Jitter éventuel sur le train seulement.

    Renvoie l'historique [(époque, perte train, perte val)].
    """
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    bce = None
    if model.mode == "supervised":
        n_pos, n_neg = float(ytr.sum()), float((ytr == 0).sum())
        pos_w = torch.tensor(min(n_neg / max(n_pos, 1.0), a.pos_weight_max), device=device)
        print(f"    pos_weight = {pos_w.item():.2f}  (train : {int(n_pos)} C2 / {int(n_neg)} bénins)")
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    gen = torch.Generator().manual_seed(RANDOM_STATE)
    jgen = torch.Generator().manual_seed(RANDOM_STATE + 1)

    def loss_of(out, target, m, y):
        return bce(out, y) if model.mode == "supervised" else recon_per_session(out, target, m).mean()

    @torch.no_grad()
    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        for i in range(0, len(Xva), 1024):
            x, m, y = Xva[i:i + 1024].to(device), Mva[i:i + 1024].to(device), yva[i:i + 1024].to(device)
            tot += loss_of(model(x, m), x, m, y).item() * len(x)
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
            inp = x
            if a.jitter_train > 0:                    # P6.4 : niveau ~ U(0, J) par session, cible = séquence propre
                lv = torch.rand(len(idx), generator=jgen) * a.jitter_train
                inp = jitter_log_dt(x, m, lv, dt_scaler[0], dt_scaler[1], a.jitter_kind, jgen)
            loss = loss_of(model(inp, m), x, m, y)
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


# --------------------------------------------------------------------------- un groupe
def run_group(group, X, mask, meta, a, device, mlflow, dt_scaler):
    Xg, Mg, g = select_group(X, mask, meta, group, a.subset, a.max_len)
    if not check_prefix_mask(Mg):
        sys.exit("[!] masque P5 non contigu : le LSTM packé suppose des pas valides en tête de séquence")
    yg = torch.from_numpy(g["label"].to_numpy().astype(np.float32))
    sp, lab = g["split"].to_numpy(), g["label"].to_numpy()
    L = Xg.shape[1]

    tr = np.flatnonzero((sp == "train") & ((lab == 0) if a.mode == "anomaly" else True))
    if a.sample_frac < 1:
        rng = np.random.default_rng(RANDOM_STATE)
        tr = np.sort(rng.choice(tr, size=max(1, int(len(tr) * a.sample_frac)), replace=False))
    va = np.flatnonzero((sp == "val") & ((lab == 0) if a.mode == "anomaly" else True))
    va_ben = np.flatnonzero((sp == "val") & (lab == 0))
    print(f"\n=== LSTM {group} ({a.mode}, {a.subset}, jitter-train {a.jitter_train}) : train {len(tr):,} "
          f"(C2 {int(lab[tr].sum()):,}), val {len(va):,}, val bénin {len(va_ben):,}, L = {L}")
    if len(va_ben) == 0 or len(tr) == 0 or len(va) == 0 or (a.mode == "supervised" and lab[tr].sum() == 0):
        print("    ! pas assez de données pour ce groupe : ignoré")
        return None

    model = SeqLSTM(X.shape[2], L, a.mode, a.hidden, a.layers, a.dropout, a.z_dim, a.bidirectional).to(device)
    t0 = time.perf_counter()
    hist = train_model(model, Xg[tr], Mg[tr], yg[tr], Xg[va], Mg[va], yg[va], a, device, dt_scaler)
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

    # P6.4 : robustesse au jitter (C2 bruités, bénin et seuil inchangés)
    jit_rows, jit_fam = [], None
    if dt_scaler is not None and a.jitter_eval:
        jit_rows, jit_fam = robustness(model, Xg, Mg, g, thr, a.jitter_eval, dt_scaler[0], dt_scaler[1],
                                       a.jitter_kind, RANDOM_STATE, device, group, clean_scores=scores)

    test = sp == "test"
    bench = benchmark_cpu(model, Xg[test] if test.any() else Xg, Mg[test] if test.any() else Mg, a.threads)
    bench.update(group=group, fit_seconds=round(fit_s, 1), epochs_run=len(hist))
    print(f"    {len(hist)} époques en {fit_s:.1f} s ; seuil = {thr:.4f} ; CPU : "
          f"{bench.get('ms_per_session_batch1')} ms/session (lot 1), {bench['size_mb']} Mo, {bench['n_params']:,} paramètres")

    if mlflow:
        mlflow.log_params({f"{group}_n_train": len(tr), f"{group}_n_val_benign": len(va_ben),
                           f"{group}_epochs_run": len(hist), f"{group}_seq_len": L})
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
        for r in jit_rows:
            v = r.get("recall")
            if v is not None and not np.isnan(v):
                mlflow.log_metric(f"jitter/{r['split']}/{group}/recall_j{round(r['jitter'] * 100)}", float(v))

    bundle = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "dims": model.dims,
              "threshold": float(thr), "history": hist}
    return dict(bundle=bundle, rows=rows, fam=fam, bench=bench, jit_rows=jit_rows, jit_fam=jit_fam)


# --------------------------------------------------------------------------- rechargement (pipeline P7 / fusion)
def load_lstm(path, device="cpu") -> dict:
    """Recharge le .pt -> {groupe: {model, threshold, mode}} (une seule fois au démarrage)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    out = {}
    for group, b in ckpt["groups"].items():
        model = SeqLSTM(**b["dims"])
        model.load_state_dict(b["state_dict"])
        model.eval()
        out[group] = {"model": model.to(device), "threshold": b["threshold"], "mode": ckpt["mode"]}
    return out


def score_lstm(loaded: dict, X: np.ndarray, mask: np.ndarray, groups, device="cpu") -> pd.DataFrame:
    """X (N, L, F) et mask (N, L) au format P5 ; `groups` = 'dns'/'web' par session.

    Renvoie lstm_score (élevé = suspect) et lstm_alert, dans l'ordre des sessions.
    """
    groups = np.asarray(groups)
    out = pd.DataFrame({"lstm_score": np.nan, "lstm_alert": False}, index=range(len(X)))
    for group, b in loaded.items():
        idx = np.flatnonzero(groups == group)
        if idx.size == 0:
            continue
        L = b["model"].seq_len
        xs = torch.from_numpy(np.ascontiguousarray(X[idx][:, :L], dtype=np.float32))
        ms = torch.from_numpy(fix_empty(np.asarray(mask[idx][:, :L], dtype=bool)))
        s = predict(b["model"], xs, ms, device)
        out.loc[idx, "lstm_score"] = s
        out.loc[idx, "lstm_alert"] = s > b["threshold"]
    return out


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p5", default=str(DEFAULT_P5), help="dossier contenant features_seq.npz (+ seq_scaler.pkl)")
    ap.add_argument("--p2", help="dossier P2 (sessions_pcap.parquet) : métriques par famille + unseen_strict")
    ap.add_argument("--mode", choices=["supervised", "anomaly"], required=True)
    ap.add_argument("--subset", choices=["all", "packets"], default="all")
    ap.add_argument("--groups", default="dns,web")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--z-dim", type=int, default=16, help="anomaly : taille du goulot (comme le Transformer)")
    ap.add_argument("--pos-weight-max", type=float, default=20.0, help="supervised : plafond du poids de la classe C2")
    ap.add_argument("--max-len", type=int, default=64, help="tronque aux N premiers pas (16 pour dns)")
    ap.add_argument("--jitter-train", type=float, default=0.0,
                    help="P6.4 : niveau max de jitter pendant l'entraînement (0,5 = 50 %%) ; niveau ~ U(0, J) par session")
    ap.add_argument("--jitter-kind", choices=JITTER_KINDS, default="sym",
                    help="sym : dt x U(1-j, 1+j) ; down : dt x U(1-j, 1) (style Cobalt Strike)")
    ap.add_argument("--jitter-eval", default=",".join(str(v) for v in DEFAULT_LEVELS),
                    help="niveaux évalués à seuil fixe sur les C2 (vide pour ne pas évaluer)")
    ap.add_argument("--sample-frac", type=float, default=1.0, help="fraction du train (test rapide)")
    ap.add_argument("--threads", type=int, default=1, help="threads CPU pour le benchmark de latence")
    ap.add_argument("--out-name", help="nom du .pt (défaut : lstm_<tag>) ; 'lstm' pour la version finale")
    ap.add_argument("--run-name", help="nom du run MLflow")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args(argv)
    a.jitter_eval = [float(v) for v in a.jitter_eval.split(",") if v.strip()]
    if not 0 <= a.jitter_train <= 1:
        ap.error("--jitter-train doit être dans [0, 1]")
    return a


def main(argv=None):
    a = parse_args(argv)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device d'entraînement : {device}  (le benchmark de latence est toujours sur CPU)")

    X, mask, meta, names = load_seq(Path(a.p5), Path(a.p2) if a.p2 else None)
    print(f"features_seq : X{X.shape}, features = {names}")
    if names[0] != "log_dt":
        sys.exit(f"[!] la 1re feature de séquence doit être log_dt (trouvé {names[0]})")
    dt_scaler = None
    if (Path(a.p5) / "seq_scaler.pkl").exists():
        dt_scaler = load_dt_scaler(a.p5)
        print(f"log_dt : moyenne {dt_scaler[0]:.3f}, écart-type {dt_scaler[1]:.3f} (seq_scaler.pkl de P5)")
    elif a.jitter_train > 0 or a.jitter_eval:
        sys.exit(f"[!] {Path(a.p5) / 'seq_scaler.pkl'} introuvable : nécessaire pour le jitter (ou --jitter-eval '')")

    tag = f"{a.mode}_{a.subset}" + (f"_jit{round(a.jitter_train * 100)}" if a.jitter_train > 0 else "")
    out_name = a.out_name or f"lstm_{tag}"
    mlflow = None if a.no_mlflow else setup_mlflow("c2-lstm")
    run = mlflow.start_run(run_name=a.run_name or f"lstm_{tag}") if mlflow else nullcontext()
    res = {}
    with run:
        if mlflow:
            mlflow.set_tags({"lot": "P6", "task": "P6.3-P6.4", "author": "wassima", "git_commit": git_commit(),
                             "features": "p5_features_tls_seq (features_seq.npz)", "mode": a.mode,
                             "subset": a.subset, "arch": "lstm"})
            mlflow.log_params({"mode": a.mode, "subset": a.subset, "hidden": a.hidden, "layers": a.layers,
                               "dropout": a.dropout, "bidirectional": a.bidirectional, "z_dim": a.z_dim,
                               "lr": a.lr, "batch": a.batch, "epochs_max": a.epochs, "patience": a.patience,
                               "max_len": a.max_len, "jitter_train": a.jitter_train, "jitter_kind": a.jitter_kind,
                               "target_fpr": TARGET_FPR, "threshold_split": "val_benign", "seed": RANDOM_STATE})
        for group in [x.strip() for x in a.groups.split(",") if x.strip()]:
            r = run_group(group, X, mask, meta, a, device, mlflow, dt_scaler)
            if r is not None:
                res[group] = r
        if not res:
            sys.exit("[!] aucun groupe entraîné")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pt = MODELS_DIR / f"{out_name}.pt"
        torch.save({"arch": "lstm", "groups": {g: r["bundle"] for g, r in res.items()}, "mode": a.mode,
                    "subset": a.subset, "jitter_train": a.jitter_train, "jitter_kind": a.jitter_kind,
                    "dt_scaler": dt_scaler, "seed": RANDOM_STATE, "target_fpr": TARGET_FPR, "features": names}, pt)
        base = EVAL_DIR / f"lstm_{tag}"
        metrics = merge_csv(Path(f"{base}_metrics.csv"), pd.DataFrame([x for r in res.values() for x in r["rows"]]))
        merge_csv(Path(f"{base}_benchmark.csv"), pd.DataFrame([r["bench"] for r in res.values()]))
        fams = [r["fam"] for r in res.values() if r["fam"] is not None]
        if fams:
            merge_csv(Path(f"{base}_by_family.csv"), pd.concat(fams, ignore_index=True))
        jit = [x for r in res.values() for x in r["jit_rows"]]
        if jit:
            merge_csv(Path(f"{base}_jitter.csv"), pd.DataFrame(jit))
        jfam = [r["jit_fam"] for r in res.values() if r["jit_fam"] is not None]
        if jfam:
            merge_csv(Path(f"{base}_jitter_by_family.csv"), pd.concat(jfam, ignore_index=True))
        if mlflow:
            mlflow.log_artifact(str(pt), artifact_path="models")
            mlflow.log_artifact(f"{base}_metrics.csv")

    print(f"\nModèle : {pt}")
    view = metrics[metrics.split.isin(["val", "test", "unseen", "unseen_strict"])][
        ["group", "split", "n_c2", "n_benign", "recall", "fpr", "roc_auc"]]
    print(view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if jit:
        j = pd.DataFrame(jit)
        print("\nRobustesse au jitter (C2 bruités, seuil fixe) :")
        print(j[j.split == "test"][["group", "jitter", "recall", "fpr"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nObjectifs : recall >= 0,98 et FPR < 0,001 (à mesurer sur le test, jamais à annoncer avant).")
    return pt


if __name__ == "__main__":
    main()
