"""P2 - tâche 5 : règles de labellisation des sessions (0 = bénin, 1 = C2, None = exclu).

Applique les règles de P1 (label_map.map_session, docs/P2_HANDOFF.md § 4) avec un seul ajustement
documenté pour DNS-EXF (voir plus bas). Contrôle croisé avec label_map.py : p2_05_compare_p1.py
(seules différences = l'ajustement DNS-EXF).

| Dataset                | -> 1 (C2)                                   | -> 0 (bénin)          | -> None (exclu)              |
|------------------------|---------------------------------------------|-----------------------|------------------------------|
| DoHBrw, pcap malveillant | session TLS vers un résolveur DoH :443     | -                     | tout le reste                |
| DoHBrw, pcap bénin     | -                                           | toutes                | -                            |
| DNS-EXF, pcap attaque  | session DNS vers cicresearch.ca (exfil)     | -                     | autres domaines (bruit Windows) |
| DNS-EXF, pcap bénin    | -                                           | toutes                | cicresearch.ca s'il apparaît |
| CTU-13 (pcaps)         | toutes (convention des auteurs)             | -                     | -                            |
| Stratosphere C2        | vers un serveur de c2_endpoints.csv         | - (jamais)            | tout le reste                |
| Stratosphere normal    | -                                           | toutes                | -                            |

Ajustement DNS-EXF : P1 labellise tout pcap heavy_*/light_* en C2. Or ces pcaps contiennent aussi le
trafic DNS normal de Windows (microsoft.com, windowsupdate.com, google...). Les labelliser C2 serait du
bruit de label ; on ne garde en C2 que le domaine d'exfiltration (cicresearch.ca, identifié dans les
quelques requêtes d'exfiltration non tronquées) et on exclut le reste.
"""
import ipaddress
from pathlib import PurePosixPath

import pandas as pd

from p2_common import C2_ENDPOINTS, DOH_RESOLVER_IPS

EXFIL_DOMAIN = "cicresearch.ca"


def load_c2_endpoints():
    """{capture: {(ip, port): tls_intercepted}} depuis meta/c2_endpoints.csv.

    Quelques lignes ont dst = "192.168.1.116,221.179.156.39" : on ne garde que les IP publiques
    (l'adresse privée est la machine infectée elle-même)."""
    ep = pd.read_csv(C2_ENDPOINTS)
    out = {}
    for row in ep.itertuples():
        for ip in str(row.dst).split(","):
            ip = ip.strip()
            if ipaddress.ip_address(ip).is_private:
                continue
            out.setdefault(row.capture, {})[(ip, int(row.port))] = bool(row.tls_intercepted)
    return out


def capture_id(rel_file):
    """stratosphere-https-c2/pcap/345-1_cobaltstrike/x.pcap -> '345-1'"""
    return PurePosixPath(rel_file).parts[2].split("_")[0]


def label_session(cat_row, session_type, server, port, parent_domain, endpoints):
    """Retourne (label, family, tls_intercepted). label : 1, 0 ou None (session à exclure)."""
    ds, cls, fam = cat_row["dataset"], cat_row["class"], cat_row["family"]

    if ds == "cira-cic-dohbrw-2020":
        to_resolver = session_type == "https" and server in DOH_RESOLVER_IPS and port == 443
        if cls == "benign":
            if session_type == "dns":
                return 0, "benign_dns", None
            return 0, ("benign_doh" if to_resolver else f"benign_{session_type}"), None
        return (1, fam, None) if to_resolver else (None, None, None)

    if ds == "cic-bell-dns-exf-2021":
        is_exfil = session_type == "dns" and parent_domain == EXFIL_DOMAIN
        if cls == "C2":
            return (1, fam, None) if is_exfil else (None, None, None)
        return (None, None, None) if is_exfil else (0, "benign_dns", None)

    if ds == "ctu13":
        return 1, fam, None

    if ds == "stratosphere-https-c2":
        hit = endpoints.get(capture_id(cat_row["file"]), {}).get((server, port))
        return (1, fam, hit) if hit is not None else (None, None, None)

    if ds == "stratosphere-normal":
        return 0, "benign_normal_user", None

    raise ValueError(f"dataset inconnu : {ds}")
