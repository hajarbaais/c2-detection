"""P4.3 - Temps inter-arrivee (IAT) / rythme, par session.

Mesure le RYTHME des echanges (le beaconing C2 est tres regulier, l'humain irregulier).
Fonctionne meme sur le trafic chiffre : on ne regarde QUE les horodatages, pas le contenu.

Sources d'horodatages selon le type de session :
  - session DNS clair -> events.parquet (1 requete = 1 ts)
  - session DoH/HTTPS -> events_pkts.parquet (1 seule connexion -> on descend au paquet,
    paquets montants direction == 'up')

Features (par session) :
  dns_iat_mean, dns_iat_median, dns_iat_std, dns_iat_cv (coefficient de variation = std/mean)
  -> CV proche de 0 = rythme d'horloge (beaconing, suspect) ; CV eleve = humain.

Usage : python -I p4_03_iat.py [chemin_dossier_output_P2]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_P2 = Path.home() / "Desktop" / "P2_livrables_v3" / "P2_livrables_v3" / "output"
P2_OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_P2
OUT_DIR = Path(__file__).resolve().parents[1] / "output"


def cv(std, mean):
    """Coefficient de variation = ecart-type / moyenne (0 si moyenne nulle)."""
    return std / mean if mean and mean > 0 else 0.0


def _self_test():
    # serie parfaitement periodique -> CV ~ 0
    per = pd.Series([0, 60, 120, 180, 240]).diff().dropna()
    assert cv(per.std(), per.mean()) < 1e-9, "periodique devrait donner CV=0"
    # serie tres irreguliere -> CV eleve
    irr = pd.Series([0, 1, 50, 52, 300]).diff().dropna()
    assert cv(irr.std(), irr.mean()) > 0.5, "serie irreguliere devrait donner CV eleve"
    print("[test] CV OK (periodique=0, irregulier eleve)")


def main():
    _self_test()

    print(f"[1/5] lecture sessions + events depuis {P2_OUT}")
    se = pd.read_parquet(P2_OUT / "sessions_pcap.parquet", columns=["session_id", "session_type", "label"])
    dns_sessions = set(se.loc[se.session_type == "dns", "session_id"])

    # (a) horodatages des sessions DNS clair -> events.parquet
    ev = pd.read_parquet(P2_OUT / "events.parquet", columns=["session_id", "event_type", "ts"])
    ts_dns = ev[(ev.event_type == "dns_query") & (ev.session_id.isin(dns_sessions))][["session_id", "ts"]]
    print(f"[2/5] {len(ts_dns):,} horodatages DNS (clair)")

    # (b) horodatages des sessions DoH/HTTPS -> events_pkts.parquet (paquets montants)
    print("[3/5] lecture events_pkts.parquet (110 Mo, patiente)...")
    pk = pd.read_parquet(P2_OUT / "events_pkts.parquet", columns=["session_id", "ts", "direction"])
    ts_web = pk[pk.direction == "up"][["session_id", "ts"]]
    print(f"       {len(ts_web):,} horodatages paquets montants (DoH/HTTPS)")

    # (c) on empile les deux sources (une session n'apparait que dans une seule)
    ts = pd.concat([ts_dns, ts_web], ignore_index=True)
    ts = ts.sort_values(["session_id", "ts"])
    ts["iat"] = ts.groupby("session_id")["ts"].diff()     # ecart avec la requete/paquet precedent
    ts = ts.dropna(subset=["iat"])                          # 1er de chaque session = pas d'ecart

    print("[4/5] agregation du rythme par session...")
    g = ts.groupby("session_id")["iat"]
    iat = pd.DataFrame({
        "dns_iat_mean": g.mean(),
        "dns_iat_median": g.median(),
        "dns_iat_std": g.std(),
    }).reset_index()
    iat["dns_iat_cv"] = [cv(s, m) for s, m in zip(iat["dns_iat_std"], iat["dns_iat_mean"])]

    # fusion dans features_dns.parquet
    feat_path = OUT_DIR / "features_dns.parquet"
    if feat_path.exists():
        base = pd.read_parquet(feat_path)
        base = base.drop(columns=[c for c in iat.columns if c != "session_id" and c in base.columns])
        merged = base.merge(iat, on="session_id", how="outer")
    else:
        merged = iat
    merged.to_parquet(feat_path, index=False)
    print(f"[5/5] {len(merged):,} sessions, {merged.shape[1]} colonnes -> {feat_path}")

    # verification : le beaconing C2 doit avoir un CV plus bas (plus regulier)
    chk = iat.merge(se, on="session_id", how="left")
    print("\nverification (CV median par classe ; bas = regulier = beaconing) :")
    for lab, name in [(0, "benin "), (1, "C2    ")]:
        s = chk[chk.label == lab]["dns_iat_cv"]
        if len(s):
            print(f"       {name} : CV median={s.median():.2f}   IAT median={chk[chk.label==lab]['dns_iat_median'].median():.2f}s   ({len(s):,} sessions)")


if __name__ == "__main__":
    main()
