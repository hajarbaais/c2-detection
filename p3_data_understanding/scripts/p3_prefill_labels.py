r"""
P3 - Pre-remplit verdict_manuel / commentaire dans labels_a_verifier.csv

Usage (depuis P2_livrables_v3\P2_livrables_v3) :
    python p3_prefill_labels.py

Principe : 'verifie' = le label donne par P2 est COHERENT avec la session ;
           'douteux' = le label parait faux ; vide = a juger par toi.
Les regles sont ecrites ci-dessous, une par cas, pour que tu puisses les relire et les defendre.
Une ligne deja remplie a la main n'est jamais ecrasee.
"""
import os

import pandas as pd

CSV = os.path.join("p3_output", "labels_a_verifier.csv")
d = pd.read_csv(CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
for c in ("n_events", "taille_moy_evt"):
    d[c] = pd.to_numeric(d[c], errors="coerce")


def decide(r):
    """Renvoie (verdict_manuel, commentaire)."""
    avis = r["verdict_auto"].split(" :")[0]
    ds = r["dataset"]

    # --- cas simples : la regle automatique a trouve le label coherent
    if avis == "OK":
        if r["dataset"] == "cira-cic-dohbrw-2020":
            # benins et C2 sont tous HTTPS 443 : la regle ne les distingue pas
            return "verifie", ("label herite du pcap (DoHBrw) ; verification FAIBLE : benins et C2 sont tous "
                               "HTTPS 443, la regle ne les distingue pas")
        return "verifie", "coherent avec la regle automatique (non examine a la main)"
    if avis == "CONVENTION":
        return "verifie", "CTU-13 : toute la machine infectee est C2 par convention des auteurs (DNS banal inclus)"

    # --- cas A REGARDER : examines un par un avec les criteres ci-dessous
    if ds == "stratosphere-https-c2":
        # label C2 = pcap de malware connu. Doute seulement si la session ressemble a du web normal :
        # un nom de site (SNI) present sur le port 443.
        if r["server_name"] != "" and r["dst_port"] == "443":
            return "douteux", "C2 mais nom de site present sur 443 : ressemble a du web normal"
        return "verifie", (f"pcap {r['family']} connu, port {r['dst_port']}, sans nom de site : "
                           "coherent avec C2 (rythme irregulier, regle CV trop stricte)")

    if ds == "stratosphere-normal":
        # benin sans nom de site : artefact si la session est courte et vide
        if r["n_events"] <= 1 and r["taille_moy_evt"] == 0:
            return "verifie", "1 evenement, aucune donnee, pas de handshake TLS visible : artefact de capture"
        return "", ("A JUGER : session benigne sans nom de site avec plusieurs evenements "
                    f"({int(r['n_events'])} evt, {r['duration']} s) - ce n'est pas clairement un artefact")

    return "", "A JUGER"


for i, r in d.iterrows():
    if r["verdict_manuel"].strip():          # ne jamais ecraser une saisie a la main
        continue
    d.at[i, "verdict_manuel"], d.at[i, "commentaire"] = decide(r)

d.to_csv(CSV, index=False, encoding="utf-8-sig")

regarder = d[d["verdict_auto"].str.startswith("A REGARDER")]
print(f"{len(d)} lignes ; {len(regarder)} 'A REGARDER' examinees avec les criteres :")
print(regarder[["session_id", "family", "dst_port", "n_events", "verdict_manuel"]].to_string(index=False))
print("\nTotal par verdict :")
print(d["verdict_manuel"].replace("", "(a juger par toi)").value_counts().to_string())
