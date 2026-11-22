"""Pourquoi le Transformer (anomaly, web) ne détecte aucune session dridex ?

Compare dridex, trickbot (75 % détecté), neris et le bénin sur les sessions web AVEC paquets réels :
  A. effectifs par split, part de http, part de « fallback », nombre de pas (longueur des séquences)
  B. valeur typique de chaque feature de séquence (en unités standardisées de P5)
  C. (avec --model) score du Transformer rapporté au seuil, et part de sessions au-dessus du seuil
  E. (avec --model) rappel par famille et FPR du bénin pour d'autres seuils (diagnostic)
  D. colonnes numériques de sessions_pcap.parquet (P2) : médiane par famille

Usage (depuis la racine du dépôt) :
  python -I p6_models/scripts/p6_06_dridex_check.py --p5 p5_features_tls_seq/output --p2 <dossier output P2> ^
         --model p6_models/models/tf_web_anomaly_packets.pt

Lecture : si dridex ressemble au bénin sur A et B, ses séquences ne contiennent pas de signal propre
(il faudra les features TLS / la fusion). Si dridex diffère mais reste sous le seuil en C, le modèle
ne l'a pas appris. Si dridex a très peu de pas, la séquence est trop courte pour un modèle séquentiel.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_05_transformer import load_seq, load_transformer, score_transformer  # noqa: E402

FAMS = ["dridex", "trickbot", "neris", "benign_https", "benign_normal_user"]
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p5", required=True, help="dossier contenant features_seq.npz")
    ap.add_argument("--p2", required=True, help="dossier P2 (sessions_pcap.parquet, colonne family)")
    ap.add_argument("--model", help="tf_web_anomaly_packets.pt (optionnel : partie C)")
    a = ap.parse_args()

    X, mask, meta, names = load_seq(Path(a.p5), Path(a.p2))
    web = meta["group"].eq("web") & meta["family"].isin(FAMS)
    m = meta[web]
    real = m[~m["fallback_only"] & ~m["empty"]]

    print("\n=== A. Effectifs et forme des séquences (web, toutes sessions) ===")
    A = m.groupby(["family", "split"]).agg(
        n=("session_id", "size"),
        part_http=("session_type", lambda s: (s == "http").mean()),
        part_fallback=("fallback_only", "mean"),
        pas_mediane=("n_steps", "median"),
        pas_p90=("n_steps", lambda s: s.quantile(0.9)),
    )
    print(A.round(3).to_string())

    print("\n=== B. Features de séquence, sessions avec paquets réels, split test (médiane des moyennes par session) ===")
    idx = real.index.to_numpy()
    Ms = mask[idx]
    cnt = Ms.sum(1).clip(min=1)
    means = (X[idx] * Ms[:, :, None]).sum(1) / cnt[:, None]
    B = pd.DataFrame(means, columns=names)
    B["family"], B["split"], B["n_pas"] = real["family"].to_numpy(), real["split"].to_numpy(), real["n_steps"].to_numpy()
    print(B[B["split"] == "test"].groupby("family")[names + ["n_pas"]].median().round(2).to_string())

    if a.model:
        print("\n=== C. Score du Transformer rapporté au seuil (test, sessions avec paquets) ===")
        loaded = load_transformer(a.model)
        thr = loaded["web"]["threshold"]
        t = real[real["split"] == "test"]
        ti = t.index.to_numpy()
        sc = score_transformer(loaded, X[ti], mask[ti], ["web"] * len(ti))
        r = pd.DataFrame({"family": t["family"].to_numpy(), "ratio": sc["transformer_score"].to_numpy() / thr})
        C = r.groupby("family")["ratio"].describe(percentiles=[0.1, 0.5, 0.9])[["count", "10%", "50%", "90%"]]
        C["part_au_dessus_du_seuil"] = r.groupby("family")["ratio"].apply(lambda s: (s > 1).mean())
        print(f"(ratio = score / seuil ; seuil = {thr:.4f} ; > 1 = alerte)")
        print(C.round(3).to_string())

        print("\n=== E. Rappel par famille selon le seuil (exploratoire : NE PAS choisir le seuil final sur le test) ===")
        ben = r[r["family"].str.startswith("benign")]
        rows = []
        for th in [0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
            row = {"seuil/seuil_actuel": th, "FPR_benin": (ben["ratio"] > th).mean()}
            for f in ["dridex", "trickbot", "neris"]:
                row[f"rappel_{f}"] = (r[r["family"] == f]["ratio"] > th).mean()
            rows.append(row)
        print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n=== D. Colonnes numériques de P2 (sessions_pcap.parquet), médiane par famille ===")
    try:
        s = pd.read_parquet(Path(a.p2) / "sessions_pcap.parquet")
        s = s[s["family"].isin(FAMS)]
        skip = {"session_id", "unseen_strict", "label"}
        num = [c for c in s.select_dtypes("number").columns if c not in skip][:30]
        print(s.groupby("family")[num].median().T.round(2).to_string())
    except Exception as e:  # noqa: BLE001
        print(f"(partie D ignorée : {e})")


if __name__ == "__main__":
    main()