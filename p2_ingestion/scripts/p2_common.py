"""P2 - constantes et chemins partagés par tous les scripts P2.

Arborescence :
    projet-c2-P2/
    ├── data/                 (livré par P1, lecture seule)
    └── p2/
        ├── scripts/          (ce dossier)
        ├── output/
        │   ├── logs/<dataset>/<pcap>/{conn,dns,ssl,x509,pkts}.parquet   (tâches 1-2)
        │   ├── sessions_pcap.parquet, events.parquet                     (tâches 3-6)
        │   └── splits/                                                   (tâche 7)
        └── docs/
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # projet-c2-P2
DATA = ROOT / "data"
EXTRACTED = DATA / "extracted"
CATALOG = DATA / "catalog_reduit.csv"
C2_ENDPOINTS = EXTRACTED / "stratosphere-https-c2" / "meta" / "c2_endpoints.csv"

OUT = ROOT / "p2" / "output"
LOGS = OUT / "logs"
SPLITS = OUT / "splits"
TMP = OUT / "tmp"

TSHARK = Path(r"C:\Program Files\Wireshark\tshark.exe")

# Ports TLS non standards utilisés par les C2 Stratosphere (c2_endpoints.csv) :
# Wireshark ne les décode pas en TLS sans "decode as".
TLS_EXTRA_PORTS = [447, 449, 3889, 4143, 4443, 4743, 8443]

# Ports « web » : sessions HTTPS (443 + ports TLS ci-dessus) et HTTP en clair (CTU-13, 2011).
HTTP_PORTS = {80, 8080}
HTTPS_PORTS = {443} | set(TLS_EXTRA_PORTS)

# Résolveurs DoH observés dans les pcaps DoHBrw malveillants (vérifié sur les 300 pcaps avec tshark :
# 1.1.1.1 x80, 176.103.130.130 x80, 9.9.9.11 x79, 8.8.4.4 x61 ; seul autre serveur : 151.101.2.49 = mise à jour système)
# + adresses sœurs des mêmes services, au cas où.
DOH_RESOLVER_IPS = {
    "1.1.1.1", "1.0.0.1",                    # Cloudflare
    "8.8.8.8", "8.8.4.4",                    # Google
    "9.9.9.9", "9.9.9.11", "149.112.112.11", "149.112.112.112",   # Quad9
    "176.103.130.130", "176.103.130.131",    # AdGuard (2020)
}
