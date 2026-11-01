"""P2 - tâches 3 à 6 : journaux -> sessions, séquences d'événements, labels, anonymisation.

Sessionisation (fiche P2, tâche 3) :
    DNS   : (IP source, domaine parent)            - domaine parent via tldextract (liste publique embarquée)
    HTTPS : (IP source, IP destination, port, SNI)  - connexions TCP avec handshake TLS ou port TLS connu
    HTTP  : (IP source, IP destination, port)       - port 80/8080 en clair (C2 HTTP des botnets CTU-13, 2011)
Chaque session est calculée À L'INTÉRIEUR d'un seul pcap, puis découpée en fenêtres de --window secondes
(défaut 3600 s = 1 h, comme les fenêtres de P4) : une balise lente sur plusieurs jours donne plusieurs
sessions, et aucune session ne mélange deux captures.

Sorties (p2/output/) :
    sessions_pcap.parquet  1 ligne par session labellisée (0/1) ; sessions « None » retirées
    events.parquet         événements ordonnés de chaque session :
                             DNS   -> 1 ligne par requête (temps, nom, type, taille, rcode, ...)
                             HTTPS -> 1 ligne par connexion TCP (temps, durée, octets montants/descendants)
    events_pkts.parquet    HTTPS/HTTP au niveau paquet (temps, sens, taille de charge utile) : utile
                           quand une session = une seule longue connexion (tunnels DoH)
    sessions_report.csv    comptes par pcap (sessions gardées / exclues par type)

Anonymisation (tâche 6) : les IP sont remplacées par sha256(sel + IP)[:16]. Le sel est dans
p2/secret_salt.txt (créé au premier lancement, à garder dans l'équipe, jamais publié).
Aucune IP en clair dans les sorties ; les IP hachées ne doivent PAS servir de feature.
"""
import argparse
import hashlib
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import tldextract

from p2_common import CATALOG, HTTP_PORTS, HTTPS_PORTS, LOGS, OUT, ROOT
from p2_labels import EXFIL_DOMAIN, label_session, load_c2_endpoints

SALT_FILE = ROOT / "p2" / "secret_salt.txt"
EXTRACT = tldextract.TLDExtract(suffix_list_urls=())      # hors ligne : liste embarquée dans le paquet
_parent_cache = {}


def get_salt():
    if not SALT_FILE.exists():
        SALT_FILE.write_text(secrets.token_hex(16), encoding="ascii")
    return SALT_FILE.read_text(encoding="ascii").strip()


def anon(series, salt):
    uniq = {ip: hashlib.sha256((salt + ip).encode()).hexdigest()[:16] for ip in series.dropna().unique()}
    return series.map(uniq)


def parent_domain(q):
    """'a.b.example.co.uk' -> 'example.co.uk' ; noms sans suffixe public (wpad, local) -> dernier label."""
    if q not in _parent_cache:
        e = EXTRACT(q)
        _parent_cache[q] = e.top_domain_under_public_suffix or q.rsplit(".", 1)[-1]
    return _parent_cache[q]


def mode(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None


def windows(df, key, window):
    """Indice de fenêtre temporelle à l'intérieur de chaque clé de session."""
    if window <= 0:
        return pd.Series(0, index=df.index)
    t0 = df.groupby(key)["ts"].transform("min")
    return ((df["ts"] - t0) // window).astype("int64")


def session_ids(file, key, win):
    raw = file + "|" + key + "|" + win.astype(str)
    return raw.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])


def dns_sessions(d, rel_file, dataset, window):
    d = d[d["query"].notna() & (d["query"] != "")].copy()
    if d.empty:
        return None, None
    q = d["query"].str.lower().str.rstrip(".")
    d["parent_domain"] = q.map(parent_domain)
    # noms tronqués par le snaplen (DNS-EXF) : la fin du nom (domaine parent) est perdue.
    # Dans DNS-EXF, toutes les requêtes tronquées sont l'exfiltration vers cicresearch.ca (vérifié :
    # les requêtes d'exfiltration non tronquées finissent toutes par ce domaine) ; ailleurs -> "<tronque>".
    trunc = d["qname_truncated"].fillna(False).astype(bool)
    d["parent_inferred"] = trunc
    d.loc[trunc, "parent_domain"] = EXFIL_DOMAIN if dataset == "cic-bell-dns-exf-2021" else "<tronque>"
    d["key"] = d["client"] + "|" + d["parent_domain"]
    d["win"] = windows(d, "key", window)
    d["session_id"] = session_ids(rel_file, d["key"], d["win"])
    g = d.groupby("session_id")
    s = pd.DataFrame({
        "session_type": "dns",
        "src_ip": g["client"].first(), "dst_ip": g["server"].agg(mode), "dst_port": 53,
        "server_name": None, "parent_domain": g["parent_domain"].first(),
        "ts_start": g["ts"].min(), "ts_end": g["ts"].max(), "n_events": g.size(),
        "qname_truncated_share": g["qname_truncated"].mean(),
        "parent_inferred": g["parent_inferred"].any(),
        "window_idx": g["win"].first(), "session_key": g["key"].first(),
    })
    ev = d[["session_id", "ts", "query", "qtype", "query_len", "resp_len", "rcode", "answers", "rtt",
            "qname_truncated"]].rename(columns={"query_len": "size", "resp_len": "resp_size"})
    ev["event_type"] = "dns_query"
    return s, ev


def web_sessions(conn, ssl, x5, pkts, rel_file, window):
    if conn.empty:
        return None, None, None
    c = conn.copy()
    if not ssl.empty:
        c = c.merge(ssl[["uid", "server_name", "ja3", "ja4", "ja3s", "version", "established"]],
                    on="uid", how="left")
        has_tls = c["ja3"].notna() | c["ja3s"].notna()
    else:
        for col in ["server_name", "ja3", "ja4", "ja3s", "version", "established"]:
            c[col] = None
        has_tls = pd.Series(False, index=c.index)
    if not x5.empty:
        c = c.merge(x5[["uid", "self_signed", "issuer_cn"]], on="uid", how="left")
    else:
        c["self_signed"], c["issuer_cn"] = None, None
    c["session_type"] = np.where(has_tls | c["resp_p"].isin(HTTPS_PORTS), "https",
                                 np.where(c["resp_p"].isin(HTTP_PORTS), "http", None))
    c = c[c["session_type"].notna()].copy()
    if c.empty:
        return None, None, None
    c["sni"] = c["server_name"].fillna("")
    c["key"] = (c["session_type"] + "|" + c["orig_h"] + "|" + c["resp_h"] + "|" + c["resp_p"].astype(str)
                + "|" + c["sni"])
    c["win"] = windows(c, "key", window)
    c["session_id"] = session_ids(rel_file, c["key"], c["win"])
    g = c.groupby("session_id")
    s = pd.DataFrame({
        "session_type": g["session_type"].first(),
        "src_ip": g["orig_h"].first(), "dst_ip": g["resp_h"].first(), "dst_port": g["resp_p"].first(),
        "server_name": g["server_name"].first(), "parent_domain": None,
        "ts_start": g["ts"].min(), "ts_end": g["ts_end"].max(), "n_events": g.size(),
        "orig_bytes": g["orig_bytes"].sum(), "resp_bytes": g["resp_bytes"].sum(),
        "orig_pkts": g["orig_pkts"].sum(), "resp_pkts": g["resp_pkts"].sum(),
        "n_conn_no_payload": ((c["orig_bytes"] + c["resp_bytes"]) == 0).groupby(c["session_id"]).sum(),
        "ja3": g["ja3"].agg(mode), "ja4": g["ja4"].agg(mode), "ja3s": g["ja3s"].agg(mode),
        "tls_version": g["version"].agg(mode),
        "cert_self_signed": g["self_signed"].agg(lambda v: bool(v.dropna().any()) if v.notna().any() else None),
        "cert_issuer_cn": g["issuer_cn"].agg(mode),
        "window_idx": g["win"].first(), "session_key": g["key"].first(),
    })
    ev = c[["session_id", "ts", "duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]].rename(
        columns={"orig_bytes": "size", "resp_bytes": "resp_size"})
    ev["event_type"] = "conn"
    pk = pkts.merge(c[["uid", "session_id"]], on="uid", how="inner")
    pk = pk[["session_id", "ts", "is_orig", "tcplen"]].rename(columns={"tcplen": "size"})
    return s, ev, pk


def process_pcap(cat_row, window, endpoints):
    rel = cat_row["file"]
    d = LOGS / Path(rel).with_suffix("")
    rd = lambda n: pd.read_parquet(d / f"{n}.parquet")
    parts_s, parts_e, parts_p = [], [], []
    dns = rd("dns")
    s, e = dns_sessions(dns, rel, cat_row["dataset"], window) if not dns.empty else (None, None)
    if s is not None:
        parts_s.append(s); parts_e.append(e)
    conn = rd("conn")
    if not conn.empty:
        s, e, p = web_sessions(conn, rd("ssl"), rd("x509"), rd("pkts"), rel, window)
        if s is not None:
            parts_s.append(s); parts_e.append(e); parts_p.append(p)
    if not parts_s:
        return None, None, None, {"file": rel}
    S = pd.concat(parts_s).reset_index(names="session_id")

    # tâche 5 : labellisation (règle évaluée une fois par combinaison distincte)
    combos = S[["session_type", "dst_ip", "dst_port", "parent_domain"]].drop_duplicates()
    res = {tuple(r): label_session(cat_row, *r, endpoints) for r in combos.itertuples(index=False)}
    lab = S[["session_type", "dst_ip", "dst_port", "parent_domain"]].apply(lambda r: res[tuple(r)], axis=1)
    S["label"] = [x[0] for x in lab]
    S["family"] = [x[1] for x in lab]
    S["tls_intercepted"] = [x[2] for x in lab]

    rep = {"file": rel}
    for t in ("dns", "https", "http"):
        st = S[S["session_type"] == t]
        rep[f"{t}_c2"] = int((st["label"] == 1).sum())
        rep[f"{t}_benign"] = int((st["label"] == 0).sum())
        rep[f"{t}_excluded"] = int(st["label"].isna().sum())
    S = S[S["label"].notna()].copy()
    keep = set(S["session_id"])
    E = pd.concat(parts_e, ignore_index=True)
    E = E[E["session_id"].isin(keep)]
    P = pd.concat(parts_p, ignore_index=True) if parts_p else None
    if P is not None:
        P = P[P["session_id"].isin(keep)]

    for col in ["dataset", "file", "capture_group", "capture_start"]:
        S[col] = cat_row[col]
    S["label"] = S["label"].astype("int8")
    S["duration"] = S["ts_end"] - S["ts_start"]
    # pcaps Stratosphere : horodatages relatifs (commencent au 01/01/1970, piège n°2 de P1)
    S["time_reference"] = "relative" if cat_row["dataset"].startswith("stratosphere") else "absolute"
    return S, E, P, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=3600, help="découpage des sessions en fenêtres (s), 0 = aucun")
    args = ap.parse_args()
    t0 = time.time()
    salt = get_salt()
    endpoints = load_c2_endpoints()
    cat = pd.read_csv(CATALOG)
    cat = cat[(cat["usable"] == "oui") & (cat["format"] == "PCAP")].reset_index(drop=True)

    S_all, E_all, P_all, reps = [], [], [], []
    for i, row in cat.iterrows():
        S, E, P, rep = process_pcap(row, args.window, endpoints)
        reps.append(rep)
        if S is None:
            continue
        S_all.append(S); E_all.append(E)
        if P is not None:
            P_all.append(P)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(cat)} pcaps", flush=True)

    S = pd.concat(S_all, ignore_index=True)
    # tâche 6 : anonymisation - remplacement des IP par leur empreinte salée
    S["src_ip_hash"] = anon(S["src_ip"], salt)
    S["dst_ip_hash"] = anon(S["dst_ip"], salt)
    S = S.drop(columns=["src_ip", "dst_ip"])
    # la clé de session contient des IP en clair : on la remplace par son empreinte
    S["session_key"] = anon(S["session_key"], salt)
    first_cols = ["session_id", "label", "family", "session_type", "dataset", "file", "capture_group",
                  "capture_start", "time_reference", "window_idx", "session_key", "src_ip_hash", "dst_ip_hash",
                  "dst_port", "server_name", "parent_domain", "ts_start", "ts_end", "duration", "n_events"]
    S = S[first_cols + [c for c in S.columns if c not in first_cols]]

    # tâche 4 : événements triés par session puis par temps, avec numéro d'ordre
    E = pd.concat(E_all, ignore_index=True).sort_values(["session_id", "ts"], kind="stable")
    E["seq"] = E.groupby("session_id").cumcount().astype("int32")
    start = S.set_index("session_id")["ts_start"]
    E["t_rel"] = E["ts"] - E["session_id"].map(start)
    E = E[["session_id", "seq", "ts", "t_rel", "event_type", "size", "resp_size", "query", "qtype", "rcode",
           "answers", "rtt", "qname_truncated", "duration", "orig_pkts", "resp_pkts"]]
    P = pd.concat(P_all, ignore_index=True).sort_values(["session_id", "ts"], kind="stable")
    P["seq"] = P.groupby("session_id").cumcount().astype("int32")
    P["direction"] = np.where(P["is_orig"], "up", "down")
    P = P[["session_id", "seq", "ts", "direction", "size"]]

    S.to_parquet(OUT / "sessions_pcap.parquet", index=False)
    E.to_parquet(OUT / "events.parquet", index=False)
    P.to_parquet(OUT / "events_pkts.parquet", index=False)
    pd.DataFrame(reps).to_csv(OUT / "sessions_report.csv", index=False)
    print(f"sessions : {len(S)}  (C2 {int((S.label == 1).sum())}, bénin {int((S.label == 0).sum())})")
    print(f"événements : {len(E)}   paquets : {len(P)}   en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
