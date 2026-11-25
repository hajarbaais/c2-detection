"""P6.4 - Diagnostic : pourquoi le jitter change (ou ne change pas) les décisions d'un modèle séquentiel.

Auteure : Wassima (sprint 3).  Répond à trois questions, sur les sessions C2 d'un split (test par défaut) :
  1. À quoi ressemblent les intervalles dt (en secondes) ?  Un jitter de 50 % sur un dt de 2 ms ne change
     presque rien après log1p ; il ne compte que sur les grands intervalles (battement du C2, en secondes).
  2. De combien bouge le score avec le jitter ?  Combien de sessions changent de décision, et combien sont
     assez proches du seuil pour pouvoir changer ?
  3. Le modèle utilise-t-il le temps ?  On remplace tous les log_dt par la moyenne du train (temps effacé)
     et on regarde le rappel ; en mode anomaly, on affiche aussi la part de l'erreur de reconstruction
     due à chaque feature.

Usage :
  python -I p6_models/scripts/p6_04b_jitter_diagnostic.py --model p6_models/models/lstm_web_anomaly_packets.pt \
         --p5 p5_features_tls_seq/output
Marche aussi avec un .pt du Transformer (p6_05). N'écrit aucun fichier.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from p6_04_jitter import DEFAULT_P5, JITTER_KINDS, jitter_log_dt, load_any, load_dt_scaler, select_group  # noqa: E402
from p6_05_transformer import load_seq, predict  # noqa: E402


def pct(v: float) -> str:
    return f"{100 * v:.1f} %"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--p5", default=str(DEFAULT_P5))
    ap.add_argument("--split", default="test")
    ap.add_argument("--level", type=float, default=0.5)
    ap.add_argument("--kind", choices=JITTER_KINDS, default="sym")
    a = ap.parse_args()

    arch, ckpt, loaded = load_any(a.model)
    X, mask, meta, names = load_seq(Path(a.p5))
    mu, sd = load_dt_scaler(a.p5)
    for group, b in loaded.items():
        model, thr = b["model"], b["threshold"]
        Xg, Mg, g = select_group(X, mask, meta, group, ckpt.get("subset", "all"), model.seq_len)
        sel = np.flatnonzero((g["split"].to_numpy() == a.split) & (g["label"].to_numpy() == 1))
        if sel.size == 0:
            print(f"[{group}] aucune session C2 dans {a.split}")
            continue
        x, m = Xg[sel], Mg[sel]
        print(f"\n=== {arch} {group} : {sel.size} sessions C2 ({a.split}), seuil {thr:.4f}")

        # 1. intervalles en secondes (pas valides, hors premier pas où dt = 0)
        dt = torch.expm1(x[:, :, 0] * sd + mu).clamp(min=0)
        first = torch.zeros_like(m)
        first[:, 0] = True
        v = dt[m & ~first].numpy()
        print(f"1. dt (s) : médiane {np.median(v):.4f} ; < 0,1 s : {pct((v < 0.1).mean())} ; "
              f"< 1 s : {pct((v < 1).mean())} ; >= 10 s : {pct((v >= 10).mean())}")
        print(f"   effet max d'un jitter {pct(a.level)} sur la valeur standardisée de log_dt : "
              f"dt = 0,01 s -> {abs(np.log1p(0.01 * (1 + a.level)) - np.log1p(0.01)) / sd:.4f} ; "
              f"dt = 60 s -> {abs(np.log1p(60 * (1 + a.level)) - np.log1p(60)) / sd:.4f}")

        # 2. mouvement des scores
        s0 = predict(model, x, m, "cpu")
        xj = jitter_log_dt(x, m, a.level, mu, sd, a.kind, torch.Generator().manual_seed(42))
        s1 = predict(model, xj, m, "cpu")
        d = np.abs(s1 - s0)
        flips = int(((s0 > thr) != (s1 > thr)).sum())
        near = int((np.abs(s0 - thr) <= d.max()).sum())
        print(f"2. |score jitter - score propre| : moyenne {d.mean():.5f}, max {d.max():.5f} "
              f"(soit {pct(np.median(d / np.maximum(np.abs(s0), 1e-12)))} du score en médiane)")
        print(f"   décisions changées : {flips} ; sessions à moins de {d.max():.5f} du seuil : {near}")
        print(f"   rappel propre {pct((s0 > thr).mean())} -> avec jitter {pct((s1 > thr).mean())}")

        # 3. le modèle utilise-t-il le temps ?
        xc = x.clone()
        xc[:, :, 0] = torch.where(m, torch.zeros_like(xc[:, :, 0]), xc[:, :, 0])   # 0 = moyenne du train
        s2 = predict(model, xc, m, "cpu")
        print(f"3. rappel avec le temps effacé (log_dt = moyenne) : {pct((s2 > thr).mean())} "
              f"(propre : {pct((s0 > thr).mean())})")
        if model.mode == "anomaly":
            with torch.no_grad():
                out = torch.cat([model(x[i:i + 1024], m[i:i + 1024]) for i in range(0, len(x), 1024)])
            err = (((out - x) ** 2) * m.unsqueeze(-1)).sum((0, 1))
            share = (err / err.sum()).numpy()
            print("   part de l'erreur de reconstruction par feature : "
                  + ", ".join(f"{n} {pct(s)}" for n, s in zip(names, share)))


if __name__ == "__main__":
    main()
