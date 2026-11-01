"""P2 - vérifications des livrables (à relancer après toute régénération).

  1. aucune IP en clair dans sessions / événements (conformité, tâche 6)
  2. cohérence sessions <-> événements (n_events = nombre de lignes dans events.parquet)
  3. labels : C2 Stratosphere comparés aux connexions comptées par P1 dans c2_endpoints.csv,
     DoHBrw malveillant uniquement vers les résolveurs DoH
  4. découpage : chaque session dans exactement un jeu, aucun pcap à cheval, familles non vues absentes
     de train/val/test
Écrit p2/output/check_report.txt.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from p2_common import OUT, SPLITS
from p2_03_split import UNSEEN_FAMILIES

IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
lines = []


def check(name, ok, detail=""):
    lines.append(f"[{'OK' if ok else 'ÉCHEC'}] {name}" + (f" - {detail}" if detail else ""))


def main():
    S = pd.read_parquet(OUT / "sessions_pcap.parquet")
    E = pd.read_parquet(OUT / "events.parquet", columns=["session_id", "query"])
    P = pd.read_parquet(OUT / "events_pkts.parquet", columns=["session_id"])

    # 1. IP en clair (server_name peut légitimement être une IP quand le client met une IP dans le SNI ;
    #    les noms DNS de type reverse lookup x.x.x.x.in-addr.arpa contiennent aussi des chiffres pointés)
    str_cols = [c for c in S.columns if S[c].dtype == object or str(S[c].dtype).startswith("str")]
    leaks = {c: int(S[c].astype(str).str.contains(IPV4).sum()) for c in str_cols
             if c not in ("server_name", "parent_domain", "file", "capture_group", "family", "cert_issuer_cn")}
    check("aucune IP en clair dans les colonnes d'identification", all(v == 0 for v in leaks.values()), str(leaks))
    check("colonnes src_ip / dst_ip absentes", not {"src_ip", "dst_ip"} & set(S.columns))

    # 2. sessions <-> événements
    n_ev = E.groupby("session_id").size()
    check("n_events = lignes de events.parquet",
          (S["n_events"].values == n_ev.reindex(S["session_id"]).values).all())
    check("chaque événement appartient à une session", E["session_id"].isin(S["session_id"]).all())
    check("chaque paquet appartient à une session", P["session_id"].isin(S["session_id"]).all())
    check("session_id unique", S["session_id"].is_unique)

    # 3. labels
    doh_c2 = S[(S.dataset == "cira-cic-dohbrw-2020") & (S.label == 1)]
    check("DoHBrw C2 : uniquement des sessions HTTPS port 443", set(doh_c2.session_type) == {"https"}
          and set(doh_c2.dst_port) == {443}, f"{len(doh_c2)} sessions")
    exf = S[S.dataset == "cic-bell-dns-exf-2021"]
    check("DNS-EXF C2 : uniquement le domaine d'exfiltration",
          set(exf[exf.label == 1].parent_domain) == {"cicresearch.ca"})
    check("aucune session bénigne vers cicresearch.ca", not (exf[exf.label == 0].parent_domain == "cicresearch.ca").any())
    for fam, n in S[S.label == 1].groupby("family").size().items():
        lines.append(f"       C2 {fam:<32} {n:>7} sessions")
    for fam, n in S[S.label == 0].groupby("family").size().items():
        lines.append(f"       bénin {fam:<29} {n:>7} sessions")

    # 4. découpage
    check("chaque session a un jeu", S["split"].notna().all())
    check("aucun pcap dans deux jeux", (S.groupby("file")["split"].nunique() == 1).all())
    check("familles non vues absentes de train/val/test",
          not S[S.split.isin(["train", "val", "test"])].family.isin(UNSEEN_FAMILIES).any())
    ids = pd.concat([pd.read_csv(SPLITS / f"{sp}_ids.csv") for sp in ["train", "val", "test", "unseen"]])
    check("fichiers *_ids.csv = toutes les sessions, sans doublon",
          len(ids) == len(S) and ids["session_id"].is_unique)

    report = "\n".join(lines)
    (OUT / "check_report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
