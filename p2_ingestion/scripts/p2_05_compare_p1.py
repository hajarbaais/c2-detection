"""P2 - contrôle croisé des labels avec label_map.py de P1 (reçu après coup dans scripts.zip).

Pour chaque session reconstruite (avant filtrage des None), compare map_session (P1) et label_session (P2).
Usage : python p2_05_compare_p1.py <dossier scripts de P1>
Écrit p2/output/compare_p1_labels.txt.
"""
import sys
from collections import Counter
from pathlib import Path

P2 = Path(__file__).resolve().parents[1]
P1 = Path(sys.argv[1])
sys.path.insert(0, str(P2 / "scripts"))
sys.path.insert(0, str(P1))

import pandas as pd
import label_map as LM
import p2_02_sessions as S2
from p2_common import CATALOG, C2_ENDPOINTS, LOGS, OUT
from p2_labels import label_session, load_c2_endpoints

LM.STRATO_ENDPOINTS_CSV = C2_ENDPOINTS
endpoints = load_c2_endpoints()
cat = pd.read_csv(CATALOG)
cat = cat[(cat["usable"] == "oui") & (cat["format"] == "PCAP")].reset_index(drop=True)

out = []
print = lambda *a: out.append(" ".join(str(x) for x in a))
agree = Counter()
diff = Counter()          # (dataset, type, p1 label, p2 label, p1 fam, p2 fam, detail) -> n sessions
errors = Counter()
for _, row in cat.iterrows():
    rel = row["file"]
    d = LOGS / Path(rel).with_suffix("")
    rd = lambda n: pd.read_parquet(d / f"{n}.parquet")
    parts = []
    dns = rd("dns")
    if not dns.empty:
        s, _ = S2.dns_sessions(dns, rel, row["dataset"], 3600)
        if s is not None:
            parts.append(s)
    conn = rd("conn")
    if not conn.empty:
        s, _, _ = S2.web_sessions(conn, rd("ssl"), rd("x509"), rd("pkts"), rel, 3600)
        if s is not None:
            parts.append(s)
    if not parts:
        continue
    S = pd.concat(parts)
    cols = ["session_type", "src_ip", "dst_ip", "dst_port", "parent_domain"]
    for combo, n in S.groupby(cols, dropna=False).size().items():
        st, src, dst, port, pdom = combo
        p2 = label_session(row, st, dst, port, pdom, endpoints)[:2]
        try:
            # relpath without the leading dataset folder, as P1 builds it
            p1 = LM.map_session(row["dataset"], rel.split("/", 1)[1], src, dst, 50000, int(port))
        except LM.UnknownLabel as e:
            errors[(row["dataset"], str(e)[:80])] += n
            continue
        if p1[0] == p2[0] and (p1[0] is None or p1[1] == p2[1]):
            agree[(row["dataset"], st)] += n
        else:
            detail = pdom if st == "dns" else f"{dst}:{port}"
            diff[(row["dataset"], st, p1[0], p2[0], p1[1], p2[1], detail if row["dataset"] != "cic-bell-dns-exf-2021" else pdom)] += n

print("AGREE (sessions):")
for k, v in sorted(agree.items()):
    print(f"  {k}: {v}")
print("\nDIFF (sessions)  dataset | type | P1 | P2 | P1 family | P2 family | detail")
for k, v in sorted(diff.items(), key=lambda kv: -kv[1])[:60]:
    print(f"  {v:>7}  {k}")
tot = Counter()
for k, v in diff.items():
    tot[k[:6]] += v
print("\nDIFF totals by (dataset,type,P1,P2,fam1,fam2):")
for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
    print(f"  {v:>7}  {k}")
print("\nERRORS:", dict(errors))
report = "\n".join(out)
(OUT / "compare_p1_labels.txt").write_text(report + "\n", encoding="utf-8")
__builtins__.print(report)
