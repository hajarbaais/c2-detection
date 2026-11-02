
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# --- dossier des sorties de P2 (1er argument, sinon le dossier livrable par defaut) ---
DEFAULT_P2 = Path.home() / "Desktop" / "P2_livrables_v3" / "P2_livrables_v3" / "output"
P2_OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_P2
OUT_DIR = Path(__file__).resolve().parents[1] / "output"


def shannon_entropy(s):
    """Entropie de Shannon (en bits) des caracteres de s. Chaine vide -> 0.0."""
    if not s:
        return 0.0
    n = len(s)
    counts = Counter(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _self_test():
    """La tache demande un test sur mot connu."""
    assert shannon_entropy("aaaa") == 0.0                 # 1 seul caractere -> 0
    assert abs(shannon_entropy("abcd") - 2.0) < 1e-9      # 4 car. distincts -> log2(4) = 2
    assert abs(shannon_entropy("aabb") - 1.0) < 1e-9      # 2 car. a 50/50 -> 1
    assert shannon_entropy("") == 0.0
    print("[test] shannon_entropy OK (aaaa=0, abcd=2, aabb=1)")


def subdomain(query, parent):
    """Partie sous-domaine = query sans le domaine parent. Ex: x9f2k.google.com -> x9f2k"""
    if not isinstance(query, str) or not query:
        return ""
    q = query.lower().rstrip(".")
    if isinstance(parent, str) and parent:
        p = parent.lower().rstrip(".")
        if q == p:
            return ""                      # requete sur le domaine parent lui-meme
        if q.endswith("." + p):
            q = q[: -(len(p) + 1)]         # on retire ".parent"
    return q.replace(".", "")              # entropie sur les caracteres, sans les points


def main():
    _self_test()

    print(f"[1/5] lecture events depuis {P2_OUT}")
    events = pd.read_parquet(P2_OUT / "events.parquet", columns=["session_id", "event_type", "query"])
    sessions = pd.read_parquet(P2_OUT / "sessions_pcap.parquet",
                               columns=["session_id", "session_type", "parent_domain", "label"])

    # [2] garder uniquement les evenements DNS
    dns = events[events["event_type"] == "dns_query"].copy()
    print(f"[2/5] {len(dns):,} requetes DNS sur {len(events):,} evenements")

    # recuperer le domaine parent de chaque session
    dns = dns.merge(sessions[["session_id", "parent_domain"]], on="session_id", how="left")

    # [3] entropie par requete, puis agregation par session
    dns["sub"] = [subdomain(q, p) for q, p in zip(dns["query"], dns["parent_domain"])]
    dns["entropy"] = dns["sub"].map(shannon_entropy)
    print("[3/5] entropie calculee par requete, agregation par session...")
    agg = dns.groupby("session_id")["entropy"].agg(dns_entropy_mean="mean", dns_entropy_max="max").reset_index()

    # [4] sauvegarde (1 ligne par session DNS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "features_dns.parquet"
    agg.to_parquet(out_path, index=False)
    print(f"[4/5] {len(agg):,} sessions DNS -> {out_path}")

    # [5] verif de bon sens : C2 doit avoir une entropie plus haute que le benin
    chk = agg.merge(sessions[["session_id", "label"]], on="session_id", how="left")
    print("[5/5] verification (entropie moyenne par classe) :")
    for lab, name in [(0, "benin "), (1, "C2    ")]:
        sub = chk[chk["label"] == lab]
        if len(sub):
            print(f"       {name} : mean={sub['dns_entropy_mean'].mean():.2f}  "
                  f"max={sub['dns_entropy_max'].mean():.2f}  ({len(sub):,} sessions)")


if __name__ == "__main__":
    main()
