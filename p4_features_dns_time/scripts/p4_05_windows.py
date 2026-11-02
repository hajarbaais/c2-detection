"""P4.5 - Fenetres longues (1 h et 6 h) : persistance et balises lentes.

P2 a decoupe les longues captures en fenetres d'1 h, reliees par `session_key`. On recolle ces
fenetres pour mesurer le comportement LONG TERME, qu'une seule heure ne montre pas :

  dns_persist_hours   : nb d'heures distinctes ou la cle est active (persistance)
  dns_span_hours      : duree totale (1ere -> derniere fenetre), en heures
  dns_coverage        : persist_hours / span -> 1.0 = actif CHAQUE heure (balise continue)
  dns_events_per_hour : debit moyen d'evenements par heure active
  dns_max_events_6h   : pic d'activite sur une fenetre de 6 h (bloc window_idx // 6)

Idee : un C2 discret est peu actif par heure MAIS present en continu pendant des heures/jours
(dridex ~120 h de suite). La continuite (coverage ~1) + le long span trahissent la balise lente.
Chaque session recoit les mesures de sa `session_key`.

Entree : sessions_pcap.parquet
Usage  : python -I p4_05_windows.py [chemin_dossier_output_P2]
"""
import sys
from pathlib import Path

import pandas as pd

DEFAULT_P2 = Path.home() / "Desktop" / "P2_livrables_v3" / "P2_livrables_v3" / "output"
P2_OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_P2
OUT_DIR = Path(__file__).resolve().parents[1] / "output"


def main():
    print(f"[1/4] lecture sessions depuis {P2_OUT}")
    se = pd.read_parquet(P2_OUT / "sessions_pcap.parquet",
                         columns=["session_id", "session_key", "window_idx", "n_events", "family", "label"])

    print("[2/4] agregation par session_key (recollage des fenetres 1 h et 6 h)...")
    g = se.groupby("session_key")
    persist = g["window_idx"].nunique()
    span = g["window_idx"].max() - g["window_idx"].min() + 1
    ev_total = g["n_events"].sum()

    # pic d'activite sur 6 h : on regroupe les heures en blocs de 6 (window_idx // 6)
    se["block6"] = se["window_idx"] // 6
    max6 = se.groupby(["session_key", "block6"])["n_events"].sum().groupby("session_key").max()

    key_feat = pd.DataFrame({
        "dns_persist_hours": persist,
        "dns_span_hours": span,
        "dns_events_per_hour": ev_total / persist,
        "dns_max_events_6h": max6,
    })
    key_feat["dns_coverage"] = key_feat["dns_persist_hours"] / key_feat["dns_span_hours"]
    key_feat = key_feat.reset_index()

    # chaque session recoit les mesures de sa cle
    out = se[["session_id", "session_key"]].merge(key_feat, on="session_key", how="left").drop(columns="session_key")

    print("[3/4] fusion dans features_dns.parquet...")
    feat_path = OUT_DIR / "features_dns.parquet"
    base = pd.read_parquet(feat_path)
    base = base.drop(columns=[c for c in out.columns if c != "session_id" and c in base.columns])
    merged = base.merge(out, on="session_id", how="left")
    merged.to_parquet(feat_path, index=False)
    print(f"[4/4] {len(merged):,} sessions, {merged.shape[1]} colonnes -> {feat_path}")

    # verification : la continuite (coverage) et le span separent-ils C2 persistant / benin ?
    chk = key_feat.merge(se.groupby("session_key").agg(family=("family", "first"), label=("label", "first")).reset_index(),
                         on="session_key")
    multi = chk[chk["dns_persist_hours"] >= 2]
    print("\nverification (cles actives >= 2 h, mediane par famille) :")
    top = multi.groupby(["label", "family"]).agg(
        n=("session_key", "size"),
        persist_h=("dns_persist_hours", "median"),
        span_h=("dns_span_hours", "median"),
        coverage=("dns_coverage", "median"),
        ev_per_h=("dns_events_per_hour", "median")).round(2)
    print(top.sort_values("persist_h", ascending=False).head(14).to_string())


if __name__ == "__main__":
    main()
