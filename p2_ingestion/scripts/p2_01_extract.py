"""P2 - tâches 1 et 2 : PCAP -> journaux conn / dns / ssl / x509 (équivalents Zeek) avec tshark.

Zeek n'étant pas disponible, chaque pcap est lu UNE fois par tshark (Wireshark 4.4), qui sort une ligne
par paquet TCP ou DNS avec les champs utiles (adresses, ports, tailles, drapeaux TCP, requête DNS,
handshake TLS, JA3/JA4/JA3S, certificat). Python reconstruit ensuite les journaux :

    conn.parquet  1 ligne par connexion TCP   (≈ conn.log)  : extrémités, durée, octets/paquets par sens
    dns.parquet   1 ligne par requête DNS     (≈ dns.log)   : nom, type, rcode, nb réponses, tailles, RTT
    ssl.parquet   1 ligne par connexion TLS   (≈ ssl.log)   : SNI, JA3, JA4, JA3S, version TLS
    x509.parquet  1 ligne par certificat feuille (≈ x509.log) : sujet, émetteur, validité, auto-signé
    pkts.parquet  paquets TCP avec charge utile des connexions web (sert aux séquences, tâche 4)

Usage :
    python -I p2/scripts/p2_01_extract.py                 # tous les pcaps du catalogue réduit
    python -I p2/scripts/p2_01_extract.py --only light_   # test sur quelques pcaps
"""
import argparse
import csv
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from cryptography import x509
from cryptography.x509.oid import NameOID

from p2_common import (CATALOG, EXTRACTED, HTTP_PORTS, HTTPS_PORTS, LOGS, OUT, TLS_EXTRA_PORTS, TMP,
                       TSHARK)

FIELDS = [
    "frame.number", "frame.time_epoch", "frame.len",
    "ip.src", "ip.dst", "ipv6.src", "ipv6.dst",
    "tcp.stream", "tcp.srcport", "tcp.dstport", "tcp.len",
    "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.fin", "tcp.flags.reset",
    "udp.srcport", "udp.dstport",
    "dns.id", "dns.flags.response", "dns.qry.name", "dns.qry.type", "dns.flags.rcode",
    "dns.count.answers", "udp.payload",
    "tls.handshake.type", "tls.handshake.extensions_server_name",
    "tls.handshake.ja3", "tls.handshake.ja4", "tls.handshake.ja3s",
    "tls.handshake.version", "tls.handshake.extensions.supported_version",
    "tls.handshake.certificate",
]
# TCP (connexions, TLS, DNS sur TCP) + DNS sur UDP/53 uniquement (pas mDNS / LLMNR / NBNS : bruit local).
DISPLAY_FILTER = "tcp or (udp.port == 53 and dns)"
LOG_NAMES = ("conn", "dns", "ssl", "x509", "pkts")

TLS_VERSIONS = {"0x0300": "SSLv3", "0x0301": "TLSv1.0", "0x0302": "TLSv1.1",
                "0x0303": "TLSv1.2", "0x0304": "TLSv1.3"}
BOOL = {"1": 1, "True": 1, "0": 0, "False": 0}


def log_dir(rel_file):
    """data/extracted/<dataset>/pcap/.../x.pcap -> p2/output/logs/<dataset>/pcap/.../x/"""
    return LOGS / Path(rel_file).with_suffix("")


def run_tshark(pcap, out_tsv):
    cmd = [str(TSHARK), "-r", str(pcap), "-n", "-Y", DISPLAY_FILTER, "-T", "fields",
           "-E", "header=y", "-E", "separator=/t", "-E", "quote=n",
           "-E", "occurrence=a", "-E", "aggregator=|"]
    for port in TLS_EXTRA_PORTS:
        cmd += ["-d", f"tcp.port=={port},tls"]
    for f in FIELDS:
        cmd += ["-e", f]
    with open(out_tsv, "wb") as fh:
        res = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE)
    # pcaps tronqués (345-1, 346-1) : tshark avertit et rend un code non nul, mais tout le reste est lu.
    return res.returncode, res.stderr.decode("utf-8", "replace").strip()[:300]


def first(s):
    """Premier élément d'un champ multi-valeurs tshark ('a|b' -> 'a')."""
    # astype("string") : une colonne entièrement vide est lue en float par pandas
    return s.astype("string").str.split("|", n=1).str[0]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def has_hs_type(s, t):
    return s.fillna("").str.split("|").apply(lambda v: t in v)


def tls_version(hex_list):
    """Plus haute version connue dans une liste '0x0304|0x0303|0x0a0a' (ignore les valeurs GREASE)."""
    if not isinstance(hex_list, str):
        return None
    known = [v for v in hex_list.split("|") if v in TLS_VERSIONS]
    return TLS_VERSIONS[max(known)] if known else None


def load_packets(tsv):
    raw = pd.read_csv(tsv, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False,
                      na_values=[""], on_bad_lines="skip")
    # paquets encapsulés (erreurs ICMP, Teredo) : plusieurs en-têtes IP -> on les écarte
    src = raw["ip.src"].fillna(raw["ipv6.src"])
    dst = raw["ip.dst"].fillna(raw["ipv6.dst"])
    keep = src.notna() & dst.notna() & ~src.str.contains("|", regex=False) & ~dst.str.contains("|", regex=False)
    raw, src, dst = raw[keep], src[keep], dst[keep]
    pk = pd.DataFrame({
        "num": num(raw["frame.number"]).astype("int64"),
        "ts": num(raw["frame.time_epoch"]),
        "flen": num(raw["frame.len"]).astype("int64"),
        "src": src, "dst": dst,
        "stream": num(first(raw["tcp.stream"])),
        "sport": num(first(raw["tcp.srcport"]).fillna(first(raw["udp.srcport"]))),
        "dport": num(first(raw["tcp.dstport"]).fillna(first(raw["udp.dstport"]))),
        "tcplen": num(first(raw["tcp.len"])).fillna(0).astype("int64"),
        "syn": first(raw["tcp.flags.syn"]).map(BOOL), "ack": first(raw["tcp.flags.ack"]).map(BOOL),
        "fin": first(raw["tcp.flags.fin"]).map(BOOL), "rst": first(raw["tcp.flags.reset"]).map(BOOL),
    })
    return raw, pk


def build_conn(pk):
    tcp = pk[pk["stream"].notna()].copy()
    if tcp.empty:
        return pd.DataFrame(), tcp
    tcp["stream"] = tcp["stream"].astype("int64")
    # initiateur = émetteur du premier SYN sans ACK ; à défaut, du premier paquet vu
    syn_only = tcp[(tcp["syn"] == 1) & (tcp["ack"] == 0)].groupby("stream").head(1)
    firsts = tcp.groupby("stream").head(1).set_index("stream")
    orig = firsts[["src", "sport", "dst", "dport"]].copy()
    s = syn_only.set_index("stream")[["src", "sport", "dst", "dport"]]
    orig.loc[s.index] = s
    orig.columns = ["orig_h", "orig_p", "resp_h", "resp_p"]
    tcp = tcp.join(orig, on="stream")
    tcp["is_orig"] = (tcp["src"] == tcp["orig_h"]) & (tcp["sport"] == tcp["orig_p"])
    tcp["o_b"] = tcp["tcplen"].where(tcp["is_orig"], 0)
    tcp["r_b"] = tcp["tcplen"].where(~tcp["is_orig"], 0)
    tcp["o_ipb"] = tcp["flen"].where(tcp["is_orig"], 0)
    tcp["r_ipb"] = tcp["flen"].where(~tcp["is_orig"], 0)
    g = tcp.groupby("stream")
    conn = pd.DataFrame({
        "ts": g["ts"].min(), "ts_end": g["ts"].max(),
        "orig_bytes": g["o_b"].sum(), "resp_bytes": g["r_b"].sum(),
        "orig_pkts": g["is_orig"].sum(), "resp_pkts": (~tcp["is_orig"]).groupby(tcp["stream"]).sum(),
        "orig_ip_bytes": g["o_ipb"].sum(), "resp_ip_bytes": g["r_ipb"].sum(),
        "syn_seen": g["syn"].max(), "fin_seen": g["fin"].max(), "rst_seen": g["rst"].max(),
    }).join(orig)
    conn["duration"] = conn["ts_end"] - conn["ts"]
    conn["orig_p"] = conn["orig_p"].astype("int64")
    conn["resp_p"] = conn["resp_p"].astype("int64")
    conn = conn.reset_index().rename(columns={"stream": "uid"})
    return conn, tcp


def parse_qname(hex_payload):
    """Décode la question d'un message DNS à partir des octets bruts (en hex).

    Les pcaps CIC-Bell-DNS-EXF-2021 sont capturés avec snaplen = 96 octets : le nom est coupé après ~42
    caractères et Wireshark refuse alors de le décoder. On garde ici la partie lisible du nom.
    Retourne (nom, tronqué ?, qtype ou None).
    """
    try:
        b = bytes.fromhex(hex_payload)
    except (TypeError, ValueError):
        return None, None, None
    i, labels = 12, []
    while i < len(b):
        n = b[i]
        if n == 0:
            qtype = int.from_bytes(b[i + 1:i + 3], "big") if i + 3 <= len(b) else None
            return ".".join(labels), False, qtype
        if n >= 0xC0:                     # pointeur de compression : impossible dans une question
            break
        labels.append(b[i + 1:i + 1 + n].decode("ascii", "replace"))
        if i + 1 + n > len(b):            # étiquette coupée par le snaplen
            return ".".join(labels), True, None
        i += 1 + n
    return (".".join(labels), True, None) if labels else (None, None, None)


def build_dns(raw, pk):
    is_dns = raw["dns.id"].notna() & ((pk["sport"] == 53) | (pk["dport"] == 53))
    d = pk[is_dns].copy()
    if d.empty:
        return pd.DataFrame()
    r = raw[is_dns]
    d["is_resp"] = first(r["dns.flags.response"]).map(BOOL)
    d["trans_id"] = first(r["dns.id"])
    d["query"] = first(r["dns.qry.name"])
    d["qtype"] = num(first(r["dns.qry.type"]))
    d["rcode"] = num(first(r["dns.flags.rcode"]))
    d["answers"] = num(first(r["dns.count.answers"]))
    d["qname_truncated"] = False
    # nom absent (paquet tronqué) -> décodage manuel des octets bruts
    miss = d["query"].isna() & r["udp.payload"].notna()
    if miss.any():
        idx = d.index[miss]
        parsed = [parse_qname(h) for h in first(r.loc[miss, "udp.payload"])]
        d.loc[idx, "query"] = pd.Series([x[0] for x in parsed], index=idx, dtype="string")
        d.loc[idx, "qname_truncated"] = pd.Series([bool(x[1]) for x in parsed], index=idx)
        d.loc[idx, "qtype"] = pd.Series([x[2] for x in parsed], index=idx, dtype="float64")
    q = d[d["is_resp"] == 0]
    # appariement requête/réponse par (client, port client, serveur, identifiant DNS), 1re réponse après la requête
    a = d[d["is_resp"] == 1][["dst", "dport", "src", "trans_id", "ts", "flen", "rcode", "answers"]]
    a.columns = ["client", "client_p", "server", "trans_id", "resp_ts", "resp_len", "rcode", "answers"]
    out = q[["num", "ts", "src", "sport", "dst", "dport", "stream", "trans_id", "query", "qtype",
             "qname_truncated", "flen"]]
    out = out.rename(columns={"src": "client", "sport": "client_p", "dst": "server", "dport": "server_p",
                              "flen": "query_len"})
    key = ["client", "client_p", "server", "trans_id"]
    pairs = out[["num", "ts"] + key].merge(a, on=key, how="inner")
    pairs = pairs[pairs["resp_ts"] >= pairs["ts"]].sort_values(["num", "resp_ts"]).drop_duplicates("num")
    m = out.merge(pairs[["num", "resp_ts", "resp_len", "rcode", "answers"]], on="num", how="left")
    m["rtt"] = m["resp_ts"] - m["ts"]
    m["proto"] = m["stream"].notna().map({True: "tcp", False: "udp"})
    return m.drop(columns=["stream", "resp_ts", "num"]).sort_values("ts").reset_index(drop=True)


def parse_cert(hex_der):
    try:
        c = x509.load_der_x509_certificate(bytes.fromhex(hex_der))
    except Exception:
        return {"cert_parse_error": True}

    def cn(name):
        v = name.get_attributes_for_oid(NameOID.COMMON_NAME)
        return v[0].value if v else None

    try:
        san = len(c.extensions.get_extension_for_class(x509.SubjectAlternativeName).value)
    except Exception:
        san = 0
    return {
        "subject": c.subject.rfc4514_string(), "issuer": c.issuer.rfc4514_string(),
        "subject_cn": cn(c.subject), "issuer_cn": cn(c.issuer),
        "not_before": c.not_valid_before_utc, "not_after": c.not_valid_after_utc,
        "self_signed": c.subject == c.issuer, "san_count": san,
        "key_type": type(c.public_key()).__name__.replace("_", "").replace("PublicKey", ""),
        "serial": format(c.serial_number, "x"), "cert_parse_error": False,
    }


def build_ssl_x509(raw, pk):
    hs = raw["tls.handshake.type"].notna() & pk["stream"].notna()
    if not hs.any():
        return pd.DataFrame(), pd.DataFrame()
    r = raw[hs]
    p = pk[hs]
    types = r["tls.handshake.type"]
    ch = has_hs_type(types, "1")
    sh = has_hs_type(types, "2")
    ce = has_hs_type(types, "11")
    client = pd.DataFrame({
        "uid": p["stream"][ch].astype("int64"), "ts": p["ts"][ch],
        "server_name": first(r["tls.handshake.extensions_server_name"][ch]),
        "ja3": first(r["tls.handshake.ja3"][ch]), "ja4": first(r["tls.handshake.ja4"][ch]),
        "client_max_version": [tls_version(a if isinstance(a, str) else b) for a, b in
                               zip(r["tls.handshake.extensions.supported_version"][ch],
                                   r["tls.handshake.version"][ch])],
    }).drop_duplicates("uid")
    server = pd.DataFrame({
        "uid": p["stream"][sh].astype("int64"),
        "ja3s": first(r["tls.handshake.ja3s"][sh]),
        # TLS 1.3 : la version choisie est dans l'extension supported_versions du ServerHello
        "version": [tls_version(a if isinstance(a, str) else b) for a, b in
                    zip(r["tls.handshake.extensions.supported_version"][sh], r["tls.handshake.version"][sh])],
    }).drop_duplicates("uid")
    ssl = client.merge(server, on="uid", how="outer")
    ssl["established"] = ssl["ja3s"].notna()

    certs = pd.DataFrame({"uid": p["stream"][ce].astype("int64"), "ts": p["ts"][ce],
                          "der": first(r["tls.handshake.certificate"][ce])}).dropna(subset=["der"])
    certs = certs.drop_duplicates("uid")
    if certs.empty:
        x5 = pd.DataFrame()
    else:
        x5 = pd.concat([certs[["uid", "ts"]].reset_index(drop=True),
                        pd.DataFrame([parse_cert(h) for h in certs["der"]])], axis=1)
    ssl["has_cert"] = ssl["uid"].isin(certs["uid"])
    return ssl, x5


def extract_one(rel_file):
    t0 = time.time()
    pcap = EXTRACTED / rel_file
    out_dir = log_dir(rel_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    tsv = TMP / (out_dir.name + f"_{abs(hash(rel_file))}.tsv")
    rc, err = run_tshark(pcap, tsv)
    try:
        raw, pk = load_packets(tsv)
        raw = raw.reset_index(drop=True)
        pk = pk.reset_index(drop=True)
        conn, tcp = build_conn(pk)
        dns = build_dns(raw, pk)
        ssl, x5 = build_ssl_x509(raw, pk)
        # paquets à charge utile des connexions web (HTTPS/HTTP par port, ou TLS détecté)
        if not tcp.empty:
            web_uids = set(conn.loc[conn["resp_p"].isin(HTTPS_PORTS | HTTP_PORTS), "uid"])
            if not ssl.empty:
                web_uids |= set(ssl["uid"])
            pk_web = tcp[(tcp["tcplen"] > 0) & tcp["stream"].isin(web_uids)]
            pkts = pk_web[["stream", "ts", "is_orig", "tcplen"]].rename(columns={"stream": "uid"})
        else:
            pkts = pd.DataFrame(columns=["uid", "ts", "is_orig", "tcplen"])
        for name, df in zip(LOG_NAMES, (conn, dns, ssl, x5, pkts)):
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
    finally:
        tsv.unlink(missing_ok=True)
    return {"file": rel_file, "tshark_rc": rc, "tshark_stderr": err, "n_packets": len(pk),
            "first_ts": pk["ts"].min(), "last_ts": pk["ts"].max(),
            "n_conn": len(conn), "n_dns": len(dns), "n_ssl": len(ssl), "n_x509": len(x5),
            "n_pkts_web": len(pkts), "seconds": round(time.time() - t0, 1)}


def done(rel_file):
    d = log_dir(rel_file)
    return all((d / f"{n}.parquet").exists() for n in LOG_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="ne traiter que les fichiers dont le chemin contient ce texte")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="refaire même si les journaux existent")
    args = ap.parse_args()

    cat = pd.read_csv(CATALOG)
    cat = cat[(cat["usable"] == "oui") & (cat["format"] == "PCAP")]
    files = cat.sort_values("size_bytes", ascending=False)["file"].tolist()
    if args.only:
        files = [f for f in files if args.only in f]
    todo = [f for f in files if args.force or not done(f)]
    print(f"{len(files)} pcaps sélectionnés, {len(todo)} à traiter", flush=True)

    report_path = OUT / "extract_report.csv"
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_one, f): f for f in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            f = futs[fut]
            try:
                row = fut.result()
                print(f"[{i}/{len(todo)}] {row['seconds']:>6}s  pkts={row['n_packets']:>8} conn={row['n_conn']:>6} "
                      f"dns={row['n_dns']:>6} ssl={row['n_ssl']:>5}  {f}", flush=True)
            except Exception:
                row = {"file": f, "error": traceback.format_exc(limit=3)}
                print(f"[{i}/{len(todo)}] ERREUR {f}\n{row['error']}", flush=True)
            rows.append(row)

    if rows:
        new = pd.DataFrame(rows)
        if report_path.exists():
            old = pd.read_csv(report_path)
            new = pd.concat([old[~old["file"].isin(new["file"])], new], ignore_index=True)
        new.to_csv(report_path, index=False)
        print(f"rapport : {report_path}")


if __name__ == "__main__":
    main()
