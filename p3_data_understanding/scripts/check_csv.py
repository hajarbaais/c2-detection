r"""
Test P3 : les CSV du CIC apportent-ils quelque chose que sessions_pcap.parquet n'a pas ?

Usage (depuis cmd) :
    python check_csv.py "C:\chemin\vers\projet-c2\data"

Le dossier doit contenir catalog.csv et extracted\<dataset>\csv\*.csv (donnees de P1).
"""
import sys
import glob
import os
import pandas as pd

# Mots-cles = informations BRUTES dont P4/P5 ont besoin (noms DNS, TLS, horodatage...)
BRUT = ["query", "qname", "domain", "ja3", "ja4", "sni", "server_name", "cert", "tls", "timestamp", "ts"]

if len(sys.argv) < 2:
    sys.exit('Usage : python check_csv.py "<dossier data de P1>"')

data = sys.argv[1]
cat_path = os.path.join(data, "catalog.csv")

# 1) Liste des CSV utilisables
if os.path.exists(cat_path):
    cat = pd.read_csv(cat_path)
    csvs = cat[(cat["format"].str.upper() == "CSV") & (cat["usable"] == "oui")]
    print("=== CSV utilisables d'apres catalog.csv ===")
    print(csvs.groupby(["dataset", "role"]).size().to_string())
    files = [os.path.join(data, "extracted", f) for f in csvs["file"]]
else:
    print("catalog.csv introuvable, recherche directe des CSV...")
    files = glob.glob(os.path.join(data, "extracted", "*", "csv", "*.csv"))

files = [f for f in files if os.path.exists(f)]
if not files:
    sys.exit("\nAUCUN CSV trouve dans : " + data +
             "\nVerifie le chemin (il doit contenir catalog.csv et extracted\\...). Aucune conclusion possible.")

# 2) Colonnes de chaque CSV (on ne lit que 5 lignes)
print("\n=== Colonnes de chaque CSV ===")
resume = []
for f in files:
    if not os.path.exists(f):
        continue
    try:
        cols = pd.read_csv(f, nrows=5, low_memory=False).columns.tolist()
    except Exception as e:
        print(f"[ERREUR] {f} : {e}")
        continue
    trouves = sorted({c for c in cols for k in BRUT if k in c.lower()})
    resume.append((os.path.basename(f), len(cols), trouves))
    print(f"\n{os.path.basename(f)}  ({len(cols)} colonnes)")
    print("  ", cols[:15], "..." if len(cols) > 15 else "")
    print("   colonnes 'brutes' trouvees :", trouves if trouves else "AUCUNE")

# 3) Conclusion
print("\n=== CONCLUSION ===")
avec = [r for r in resume if r[2]]
if not avec:
    print("Aucun CSV ne contient de noms DNS / JA3 / SNI : ce sont des features calculees.")
    print("-> Preuve : pas besoin de fusionner.")
else:
    print("Ces CSV ont des colonnes 'brutes' a examiner a la main :")
    for nom, n, t in avec:
        print(f"  {nom} : {t}")
    print("-> Regarde si ces colonnes manquent vraiment dans sessions_pcap.parquet.")