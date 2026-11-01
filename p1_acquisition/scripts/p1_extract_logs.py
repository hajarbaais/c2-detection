"""P2 - extraction des journaux à faire chez P1 (qui a les PCAP complets), pour éviter de transférer ~8,6 Go.

Lance la même extraction que p2_01_extract.py (tâches 1-2) sur les PCAP benign complets ajoutés au test,
puis zippe les journaux (~1 % de la taille des PCAP) dans logs_P2.zip, à renvoyer à P2.

Les PCAP de plus de --chunk-gb Go sont découpés avec editcap (pour ne pas saturer la RAM), extraits morceau
par morceau, puis les journaux sont recollés : un seul dossier par PCAP, comme pour les autres.

Usage :
    python p1_extract_logs.py --pcaps "D:\\chemin\\vers\\dossier_des_pcap"
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import p2_01_extract as X

# chemin dans le catalogue P1 -> sha256 du fichier complet (catalog_reduit.csv)
TARGETS = {
    "cira-cic-dohbrw-2020/pcap/benign/chrome-quad9/dump_00009_20200113074803.pcap":
        "715c139d2c071166ba1fcea3cfe8326e41d56f23a39ce0cc2ff1b2bf8b76e4bf",
    "cira-cic-dohbrw-2020/pcap/benign/chrome-google/dump_00005_20200113142226.pcap":
        "073f1a9d2bf0f1bbfe41aaeab9ff250c358795a719f1f12dadf8897c80d84863",
    "cira-cic-dohbrw-2020/pcap/benign/chrome-cloudflare/dump_00005_20200113205730.pcap":
        "413b63e7db1c35b4918b1e4fe5e07999aace2c685895816250711fc2bfd82cb2",
    "cira-cic-dohbrw-2020/pcap/benign/chrome-adguard/dump_00005_20200114153502.pcap":
        "f0cc67c14b00459260c4910b40d7e3dcf953b147640768651257163b54ce157a",
    "cira-cic-dohbrw-2020/pcap/benign/firefox-google/large/dump.pcap":
        "2994e22e501afa110900862081d0c9c6c6b09272dfbcaf3eb73582f282694467",
    "stratosphere-normal/pcap/normal-14/2017-07-23_capture-winFull.pcap":
        "76317977dd5eeea4af1f9cc1ebf2fa6887d241e03cf7c8eeffd3bbcdea9db7f9",
    # lot 2 (pour approcher 100 000 sessions bénignes en test)
    "cira-cic-dohbrw-2020/pcap/benign/firefox-adguard/dump.pcap":
        "91f6d1bfbb2f714f9ffe60ce66223211c27f39e8d067176c281aa8ef64068642",
    "cira-cic-dohbrw-2020/pcap/benign/firefox-cloudflare/dump.pcap":
        "3ac780b5c1aaa4396266e77dd047c252e401fe037a4241f282dfd70fe19c3015",
    "cira-cic-dohbrw-2020/pcap/benign/firefox-quad9/dump.pcap":
        "14c9af3366cd2c264dbf3c946213264ee4b7327ad7d557eed03d380a2bb1678a",
}
UID_OFFSET = 10_000_000          # uid = n° de morceau * UID_OFFSET + tcp.stream (évite les doublons entre morceaux)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_pcap(root, rel):
    """Cherche le fichier par la fin de son chemin (le dossier de P1 n'a pas forcément la même racine)."""
    parts = Path(rel).parts
    # firefox-google/{small,medium,large}/dump.pcap : il faut le dossier parent en plus
    tail = parts[-3:] if parts[-2] in ("small", "medium", "large") else parts[-2:]
    hits = [p for p in root.rglob(parts[-1]) if p.parts[-len(tail):] == tail]
    # s'il y a aussi une version tronquée du même fichier, prendre la plus grosse
    return max(hits, key=lambda p: p.stat().st_size) if hits else None


def extract_logs(pcap, tmp):
    """Même travail que X.extract_one, mais sur un chemin quelconque ; renvoie les 5 tables."""
    tsv = tmp / f"{pcap.stem}.tsv"
    rc, err = X.run_tshark(pcap, tsv)
    try:
        raw, pk = X.load_packets(tsv)
        raw, pk = raw.reset_index(drop=True), pk.reset_index(drop=True)
        conn, tcp = X.build_conn(pk)
        dns = X.build_dns(raw, pk)
        ssl, x5 = X.build_ssl_x509(raw, pk)
        if not tcp.empty:
            web_uids = set(conn.loc[conn["resp_p"].isin(X.HTTPS_PORTS | X.HTTP_PORTS), "uid"])
            if not ssl.empty:
                web_uids |= set(ssl["uid"])
            pk_web = tcp[(tcp["tcplen"] > 0) & tcp["stream"].isin(web_uids)]
            pkts = pk_web[["stream", "ts", "is_orig", "tcplen"]].rename(columns={"stream": "uid"})
        else:
            pkts = pd.DataFrame(columns=["uid", "ts", "is_orig", "tcplen"])
    finally:
        tsv.unlink(missing_ok=True)
    return dict(zip(X.LOG_NAMES, (conn, dns, ssl, x5, pkts))), rc, err, len(pk)


def process(pcap, out_dir, tmp, chunk_bytes, chunk_pkts):
    if pcap.stat().st_size <= chunk_bytes:
        parts = [pcap]
    else:
        cdir = tmp / f"{pcap.stem}_chunks"
        cdir.mkdir(parents=True, exist_ok=True)
        editcap = X.TSHARK.with_name("editcap.exe")
        subprocess.run([str(editcap), "-c", str(chunk_pkts), str(pcap), str(cdir / "part.pcap")], check=True)
        parts = sorted(cdir.iterdir())
        print(f"    découpé en {len(parts)} morceaux", flush=True)
    tables = {n: [] for n in X.LOG_NAMES}
    n_pkts = 0
    for i, part in enumerate(parts):
        logs, rc, err, n = extract_logs(part, tmp)
        n_pkts += n
        for name, df in logs.items():
            if not df.empty and "uid" in df.columns:
                df = df.assign(uid=df["uid"].astype("int64") + i * UID_OFFSET)
            tables[name].append(df)
        if part != pcap:
            part.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, dfs in tables.items():
        dfs = [d for d in dfs if not d.empty]
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    return n_pkts, {n: sum(len(d) for d in dfs) for n, dfs in tables.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcaps", required=True, help="dossier qui contient les PCAP téléchargés (recherche récursive)")
    ap.add_argument("--out", default="logs_P2", help="dossier de sortie (défaut : logs_P2)")
    ap.add_argument("--tshark", default=str(X.TSHARK), help="chemin de tshark.exe")
    ap.add_argument("--chunk-gb", type=float, default=1.2, help="découper les PCAP plus gros que ça (Go)")
    ap.add_argument("--chunk-pkts", type=int, default=1_500_000, help="paquets par morceau")
    ap.add_argument("--skip-sha", action="store_true", help="ne pas vérifier les sha256 (plus rapide)")
    ap.add_argument("--only", help="ne traiter que les fichiers dont le chemin contient un de ces textes (séparés par des virgules)")
    args = ap.parse_args()

    X.TSHARK = Path(args.tshark)
    if not X.TSHARK.exists():
        sys.exit(f"tshark introuvable : {X.TSHARK}  (installer Wireshark ou passer --tshark)")
    root, out = Path(args.pcaps), Path(args.out).resolve()
    tmp = out.parent / (out.name + "_tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    rows = []
    only = [o.strip() for o in args.only.split(",")] if args.only else None
    targets = {r: d for r, d in TARGETS.items() if not only or any(o in r for o in only)}
    for rel, digest in targets.items():
        pcap = find_pcap(root, rel)
        if pcap is None:
            print(f"[absent] {rel}", flush=True)
            rows.append({"file": rel, "status": "absent"})
            continue
        if not args.skip_sha:
            print(f"[sha256] {pcap}", flush=True)
            if sha256(pcap) != digest:
                print("    ATTENTION : sha256 différent du catalogue (fichier incomplet ?) -> ignoré", flush=True)
                rows.append({"file": rel, "status": "sha256 différent"})
                continue
        t0 = time.time()
        print(f"[extraction] {rel}  ({pcap.stat().st_size / 1e9:.2f} Go)", flush=True)
        n_pkts, counts = process(pcap, out / "logs" / Path(rel).with_suffix(""), tmp,
                                 int(args.chunk_gb * 1e9), args.chunk_pkts)
        sec = round(time.time() - t0, 1)
        print(f"    {sec}s  paquets={n_pkts}  " + "  ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
        rows.append({"file": rel, "status": "ok", "n_packets": n_pkts, **{f"n_{k}": v for k, v in counts.items()},
                     "seconds": sec})

    shutil.rmtree(tmp, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "p1_extract_report.csv", index=False)
    zpath = out.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out.parent))
    print(f"\nTerminé. À envoyer à P2 : {zpath}  ({zpath.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
