"""P4.2 - Forme des requetes DNS, par session.

Features (par session) :
  dns_qname_len_mean / _max  : longueur du nom (caracteres)
  dns_n_levels_mean          : nombre de niveaux (labels separes par des points)
  dns_digit_ratio_mean       : part de chiffres dans le nom (encodage)
  dns_pct_txt / _cname / _null : part des types rares (tunnels)   -> qtype 16 / 5 / 10
  dns_unique_ratio           : part de requetes differentes (exfil = tout unique)
  dns_pct_nxdomain           : part de NXDOMAIN (rcode == 3)

Entree : events.parquet (query, qtype, rcode)
Sortie : ajoute ces colonnes a features_dns.parquet (1 ligne par session).

Note (P3) : ne jamais utiliser qname_truncated comme feature. Les noms DNS-EXF sont
tronques (~42 car.) mais restent bien plus longs que les sous-domaines normaux.

Usage : python -I p4_02_shape.py [chemin_dossier_output_P2]
"""
import sys
from pathlib import Path

import pandas as pd

DEFAULT_P2 = Path.home() / "Desktop" / "P2_livrables_v3" / "P2_livrables_v3" / "output"
P2_OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_P2
OUT_DIR = Path(__file__).resolve().parents[1] / "output"

QTYPE_TXT, QTYPE_CNAME, QTYPE_NULL = 16, 5, 10
RCODE_NXDOMAIN = 3


def digit_ratio(s):
    """Part de chiffres parmi les caracteres alphanumeriques du nom."""
    if not isinstance(s, str) or not s:
        return 0.0
    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    tot = digits + letters
    return digits / tot if tot else 0.0


def main():
    print(f"[1/4] lecture events depuis {P2_OUT}")
    ev = pd.read_parquet(P2_OUT / "events.parquet",
                         columns=["session_id", "event_type", "query", "qtype", "rcode"])
    dns = ev[ev["event_type"] == "dns_query"].copy()
    print(f"[2/4] {len(dns):,} requetes DNS -> calcul par requete")

    q = dns["query"].astype("string").fillna("")
    dns["qlen"] = q.str.len()
    dns["levels"] = q.str.count(r"\.") + 1
    dns["dratio"] = [digit_ratio(x) for x in q]
    dns["is_txt"] = (dns["qtype"] == QTYPE_TXT)
    dns["is_cname"] = (dns["qtype"] == QTYPE_CNAME)
    dns["is_null"] = (dns["qtype"] == QTYPE_NULL)
    dns["is_nx"] = (dns["rcode"] == RCODE_NXDOMAIN)

    print("[3/4] agregation par session...")
    g = dns.groupby("session_id")
    shape = pd.DataFrame({
        "dns_qname_len_mean": g["qlen"].mean(),
        "dns_qname_len_max": g["qlen"].max(),
        "dns_n_levels_mean": g["levels"].mean(),
        "dns_digit_ratio_mean": g["dratio"].mean(),
        "dns_pct_txt": g["is_txt"].mean(),
        "dns_pct_cname": g["is_cname"].mean(),
        "dns_pct_null": g["is_null"].mean(),
        "dns_unique_ratio": g["query"].nunique() / g["query"].size(),
        "dns_pct_nxdomain": g["is_nx"].mean(),
    }).reset_index()

    # fusion avec les features P4.1 deja presentes
    feat_path = OUT_DIR / "features_dns.parquet"
    if feat_path.exists():
        base = pd.read_parquet(feat_path)
        base = base.drop(columns=[c for c in shape.columns if c != "session_id" and c in base.columns])
        merged = base.merge(shape, on="session_id", how="outer")
    else:
        merged = shape
    merged.to_parquet(feat_path, index=False)
    print(f"[4/4] {len(merged):,} sessions, {merged.shape[1]} colonnes -> {feat_path}")

    # verification : exfil vs benin
    se = pd.read_parquet(P2_OUT / "sessions_pcap.parquet", columns=["session_id", "label"])
    chk = merged.merge(se, on="session_id", how="left")
    cols = ["dns_qname_len_mean", "dns_digit_ratio_mean", "dns_pct_txt", "dns_pct_null",
            "dns_unique_ratio", "dns_pct_nxdomain"]
    print("\nverification (moyenne par classe) :")
    print(f"{'feature':24} {'benin':>8} {'C2':>8}")
    for c in cols:
        b = chk.loc[chk.label == 0, c].mean()
        m = chk.loc[chk.label == 1, c].mean()
        print(f"{c:24} {b:8.2f} {m:8.2f}")


if __name__ == "__main__":
    main()
