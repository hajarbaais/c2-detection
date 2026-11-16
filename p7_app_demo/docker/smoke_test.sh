#!/usr/bin/env bash
# P7.6 - Test de fumée de l'image : versions, plugins JA3/JA4, imports Python, Zeek sur un PCAP.
# Usage dans le conteneur : bash p7_app_demo/docker/smoke_test.sh [chemin/vers/capture.pcap]
set -u
PCAP="${1:-${PCAP:-}}"
OUT="${ZEEK_LOG_DIR:-/output}/smoke"
ok=0; ko=0
check() { if eval "$2" >/dev/null 2>&1; then echo "[OK ] $1"; ok=$((ok+1)); else echo "[KO ] $1"; ko=$((ko+1)); fi; }

echo "== Versions"
zeek --version
echo "Plugins Zeek (liste figée au build, /opt/zeek-packages.txt) :"
cat /opt/zeek-packages.txt
tshark --version | head -1
python --version
python - <<'PY'
import importlib
for m in ("torch", "pandas", "numpy", "pyarrow", "sklearn", "joblib"):
    try:
        print(f"{m:10s}", importlib.import_module(m).__version__)
    except Exception as e:
        print(f"{m:10s} ERREUR : {e}")
PY

echo; echo "== Contrôles"
check "zeek présent"            "command -v zeek"
check "tshark présent"          "command -v tshark"
check "plugin ja3 installé"     "grep -qi ja3 /opt/zeek-packages.txt"
check "plugin ja4 installé"     "grep -qi ja4 /opt/zeek-packages.txt"
check "imports Python (torch, pandas, numpy, pyarrow, sklearn, joblib)" \
      "python -c 'import torch, pandas, numpy, pyarrow, sklearn, joblib'"
check "exécution non root"      "[ \"\$(id -u)\" != 0 ]"
if [ -f /app/p6_models/models/autoencoder.pt ]; then
  check "autoencoder.pt se recharge" \
        "python -c 'from p5_6b_autoencoder_train import load_autoencoder; load_autoencoder(\"/app/p6_models/models/autoencoder.pt\")'"
else
  echo "[--] autoencoder.pt absent (monter p6_models/models en volume pour tester son chargement)"
fi

echo; echo "== Zeek sur un PCAP"
if [ -z "$PCAP" ]; then
  PCAP=$(find / -type f \( -iname "*.pcap" -o -iname "*.pcapng" \) -path "*ja4*" 2>/dev/null | head -1)
  [ -n "$PCAP" ] && echo "PCAP de test trouvé dans le paquet ja4 : $PCAP"
fi
if [ -n "$PCAP" ] && [ -f "$PCAP" ]; then
  rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT" || exit 1
  zeek -C -r "$PCAP" packages >zeek_stdout.txt 2>&1
  if ! grep -q "^#fields.*ja4" ssl.log 2>/dev/null; then zeek -C -r "$PCAP" ja3 ja4 >>zeek_stdout.txt 2>&1; fi
  check "ssl.log produit"        "[ -s ssl.log ]"
  check "colonne ja3 dans ssl.log" "grep '^#fields' ssl.log | grep -qw ja3"
  check "colonne ja4 dans ssl.log" "grep '^#fields' ssl.log | grep -qw ja4"
  echo "Journaux produits dans $OUT :"; ls -1 *.log 2>/dev/null
  echo "Exemples (ja3, ja4) :"; zeek-cut ja3 ja4 < ssl.log 2>/dev/null | sort -u | head -3
else
  echo "[--] aucun PCAP fourni : donner un fichier avec PCAP=/data/pcaps/xxx.pcap"
  ko=$((ko+1))
fi

echo; echo "Résultat : $ok OK, $ko KO"
[ "$ko" -eq 0 ]
