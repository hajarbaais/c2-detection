"""P6.6 (préparation) - Exporte le score du LSTM de Wassima pour CHAQUE session, dans un fichier parquet.

Le LSTM `lstm_web_anomaly_packets.pt` n'existe que pour le web, et n'est valable que pour les sessions qui ont de
vrais paquets (pas les sessions « fallback » ni vides). Les autres sessions (DNS, fallback) reçoivent un score vide.

À LANCER depuis le code de la branche `feature/wassima-sprint3` (dossier extrait de wassima-sprint3-code.zip) :
ce script importe p6_03_lstm.py et p6_05_transformer.py qui se trouvent à côté de lui.
Copier ce fichier dans  wassima-sprint3/p6_models/scripts/ .

Usage :
  python -I p6_models/scripts/p6_07a_export_lstm_scores.py \
      --p5 <partage_wassima>/p5_features_tls_seq/output      (features_seq.npz + seq_scaler.pkl DE WASSIMA)
      --p2 <dossier contenant sessions_pcap.parquet> \
      --model <partage_wassima>/p6_models/models/lstm_web_anomaly_packets.pt \
      --out lstm_scores.parquet

Contrôle : sur le test web, le script doit retrouver les chiffres du README de Wassima
(1 874 C2 / 75 103 bénins ; rappel 42,16 % ; FPR 0,29 % ; 218 fausses alertes).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "evaluation"))
from p6_03_lstm import load_lstm, score_lstm  # noqa: E402
from p6_05_transformer import load_seq  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p5", required=True, help="dossier contenant features_seq.npz et seq_scaler.pkl")
    ap.add_argument("--p2", required=True, help="dossier contenant sessions_pcap.parquet")
    ap.add_argument("--model", required=True, help="fichier .pt du LSTM (lstm_web_anomaly_packets.pt)")
    ap.add_argument("--out", default="lstm_scores.parquet")
    a = ap.parse_args()

    X, mask, meta, _ = load_seq(Path(a.p5), Path(a.p2))
    loaded = load_lstm(a.model)
    res = score_lstm(loaded, X, mask, meta["group"].to_numpy())
    meta["lstm_score"] = res["lstm_score"].to_numpy()

    # sessions pour lesquelles le modèle « packets » est valable
    valid = ((meta["group"] == "web") & ~meta["fallback_only"] & ~meta["empty"]).to_numpy()
    meta.loc[~valid, "lstm_score"] = np.nan
    thr = {g: float(b["threshold"]) for g, b in loaded.items()}
    meta["lstm_threshold"] = meta["group"].map(thr)
    meta["lstm_alert"] = (meta["lstm_score"] > meta["lstm_threshold"]).fillna(False)

    keep = ["session_id", "group", "split", "label", "family", "unseen_strict", "session_type", "n_steps",
            "fallback_only", "lstm_score", "lstm_threshold", "lstm_alert"]
    meta[keep].to_parquet(a.out, index=False)

    print(f"{len(meta):,} sessions ; scores valides : {int(valid.sum()):,} (web avec paquets) ; groupes du modèle : {list(loaded)}")
    t = meta[(meta["group"] == "web") & (meta["split"] == "test") & valid]
    c2, ben = t[t["label"] == 1], t[t["label"] == 0]
    print(f"Contrôle (web, test, sessions avec paquets) : {len(c2):,} C2 / {len(ben):,} bénins ; "
          f"rappel {c2['lstm_alert'].mean():.2%} ; FPR {ben['lstm_alert'].mean():.2%} ({int(ben['lstm_alert'].sum())} fausses alertes)")
    print("Attendu d'après le README de Wassima : 1 874 C2 / 75 103 bénins ; rappel 42,16 % ; FPR 0,29 % ; 218 fausses alertes")
    print(f"Fichier écrit : {a.out}")


if __name__ == "__main__":
    main()
