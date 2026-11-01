r"""
P3 - Suite du controle qualite

Usage (depuis le dossier P2_livrables_v3\P2_livrables_v3, apres quality.py) :
    python p3_suite.py

Fait 3 choses :
  A. termine la partie 6 (share_no_payload, ctu13_broad_label)
  B. tire un echantillon de sessions pour verifier les labels (fichier CSV a remplir)
  C. dessine les graphiques de controle (PNG)

Sorties dans p3_output\ :
  suite_report.txt, labels_a_verifier.csv, labels_summary.txt, graphs\*.png
"""
import math
import os
import sys
from collections import Counter

import pandas as pd

OUT = "p3_output"
UNIFIED = os.path.join(OUT, "sessions_unified.parquet")
EVENTS = os.path.join("output", "events.parquet")
N_PAR_GROUPE = 20      # sessions tirees par (dataset, famille)
SEED = 42              # graine fixe = tirage reproductible

if not os.path.exists(UNIFIED):
    sys.exit("Lance d'abord : python quality.py")

os.makedirs(os.path.join(OUT, "graphs"), exist_ok=True)
df = pd.read_parquet(UNIFIED)

report = []


def say(t=""):
    print(t)
    report.append(str(t))


def title(t):
    say("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# ======================================================== A. FIN DE LA PARTIE 6
title("A1. share_no_payload : part de connexions sans donnees (HTTPS/HTTP)")
web = df[df["session_type"].isin(["https", "http"])]
say(web.groupby(["dataset", "label"])["share_no_payload"]
    .agg(["count", "mean", "median"]).round(3).to_string())
say("\nLecture : si l'ecart bénin/C2 n'existe que dans CTU-13, ce n'est pas un signal general.")

title("A2. ctu13_broad_label : DNS banal de la machine infectee (CTU-13)")
c = df[df["dataset"] == "ctu13"]
say("Sessions CTU-13 par type :")
say(c.groupby("session_type").size().to_string())
say(f"\nSessions DNS CTU-13 marquees C2 : {int(df['ctu13_broad_label'].sum())}")
say("Domaines les plus frequents (ce sont des sites ordinaires = DNS banal) :")
say(df[df["ctu13_broad_label"]]["parent_domain"].value_counts().head(10).to_string())
dns_c2 = df[(df["session_type"] == "dns") & (df["label"] == 1)]
say("\nDNS C2 : combien viennent de CTU-13 et combien de vrais tunnels (DNS-EXF) ?")
say(dns_c2.groupby("dataset").size().to_string())
say("\nDNS C2 dans le jeu TEST, avec et sans CTU-13 :")
t = dns_c2[dns_c2["split"] == "test"]
say(f"  avec CTU-13 : {len(t)}   sans CTU-13 : {int((~t['ctu13_broad_label']).sum())}")

# ======================================================= B. LABELS A VERIFIER
title("B. ECHANTILLON POUR VERIFIER LES LABELS")
# melange reproductible (graine fixe), puis on garde les N premieres lignes de chaque groupe
# (pas de groupby.apply : avec pandas 3 il retire les colonnes de regroupement)
sample = (df.sample(frac=1, random_state=SEED)
          .groupby(["dataset", "family"], sort=False).head(N_PAR_GROUPE)
          .reset_index(drop=True))
say(f"{len(sample)} sessions tirees ({N_PAR_GROUPE} max par dataset x famille, graine {SEED})")

ids = sample["session_id"].tolist()
ev = pd.read_parquet(EVENTS, filters=[("session_id", "in", ids)])
say(f"{len(ev)} evenements lus pour ces sessions")


def entropie(s):
    if not isinstance(s, str) or not s:
        return float("nan")
    n = len(s)
    return -sum(k / n * math.log2(k / n) for k in Counter(s).values())


def premier_label(q):
    return q.split(".")[0] if isinstance(q, str) else ""


stats = {}
for sid, g in ev.groupby("session_id"):
    g = g.sort_values("seq")
    d = g["t_rel"].diff().dropna()
    cv = d.std() / d.mean() if len(d) >= 2 and d.mean() > 0 else float("nan")
    ent = g["query"].map(premier_label).map(entropie).mean() if g["query"].notna().any() else float("nan")
    stats[sid] = {"cv_intervalles": round(cv, 3) if cv == cv else None,
                  "entropie_moy": round(ent, 2) if ent == ent else None,
                  "taille_moy_evt": round(g["size"].mean(), 1)}
st = pd.DataFrame.from_dict(stats, orient="index")
sample = sample.merge(st, left_on="session_id", right_index=True, how="left")


def verdict(r):
    """Avis automatique a confirmer a la main. 'OK ...' ou 'A REGARDER ...'."""
    ds, lab = r["dataset"], r["label"]
    if ds == "cic-bell-dns-exf-2021":
        exf = r["parent_domain"] == "cicresearch.ca"
        if lab == 1 and exf:
            return "OK : domaine d'exfiltration cicresearch.ca"
        if lab == 0 and not exf:
            return "OK : pcap benin, autre domaine"
        return "A REGARDER : label et domaine incoherents"
    if ds == "cira-cic-dohbrw-2020":
        if lab == 1:
            ok = r["session_type"] == "https" and r["dst_port"] == 443
            return "OK : HTTPS 443 d'un pcap malveillant" if ok else "A REGARDER : pas HTTPS 443"
        return "OK : pcap benin"
    if ds == "ctu13":
        return "CONVENTION : toute la machine infectee est C2 (a juger : DNS banal ?)" \
            if r["session_type"] == "dns" else "OK : trafic de la machine infectee"
    if ds == "stratosphere-https-c2":
        cv, n = r["cv_intervalles"], r["n_events"]
        if n >= 3 and cv is not None and cv <= 1:
            return f"OK : balisage regulier (CV={cv})"
        return "A REGARDER : peu d'evenements ou rythme irregulier"
    if ds == "stratosphere-normal":
        # HTTP en clair (port 80) n'a pas de SNI : ce n'est pas une anomalie
        if r["session_type"] in ("dns", "http") or isinstance(r["server_name"], str):
            return "OK : machine propre, site identifie ou HTTP sans SNI"
        return "A REGARDER : HTTPS benin sans nom de site"
    return "A REGARDER : dataset inconnu"


sample["verdict_auto"] = sample.apply(verdict, axis=1)
sample["verdict_manuel"] = ""
sample["commentaire"] = ""
cols = ["session_id", "dataset", "family", "label", "session_type", "split", "parent_domain",
        "server_name", "dst_port", "n_events", "duration", "cv_intervalles", "entropie_moy",
        "taille_moy_evt", "verdict_auto", "verdict_manuel", "commentaire"]
sample[cols].sort_values(["dataset", "family"]).to_csv(
    os.path.join(OUT, "labels_a_verifier.csv"), index=False, encoding="utf-8-sig")

sample["avis"] = sample["verdict_auto"].str.split(" :").str[0]
summ = sample.groupby(["dataset", "avis"]).size().unstack(fill_value=0)
say("\nAvis automatique par dataset :")
say(summ.to_string())
say("\nOK = coherent. A REGARDER / CONVENTION = a juger par toi dans le CSV.")
with open(os.path.join(OUT, "labels_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summ.to_string())

# =============================================================== C. GRAPHIQUES
title("C. GRAPHIQUES")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    say("matplotlib absent. Installe-le puis relance :  python -m pip install matplotlib")
    plt = None

if plt:
    import numpy as np

    G = os.path.join(OUT, "graphs")
    nom = {0: "benin", 1: "C2"}
    coul = {0: "tab:blue", 1: "tab:red"}

    def hist_log(col, fname, titre, data=df, bins=60):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for lab in (0, 1):
            v = data.loc[data["label"] == lab, col].dropna()
            v = v[v > 0]
            if len(v):
                ax.hist(np.log10(v), bins=bins, alpha=0.55, color=coul[lab],
                        label=f"{nom[lab]} (n={len(v)})")
        ax.set_xlabel(f"log10({col})  (1 = 10, 2 = 100, 3 = 1000...)")
        ax.set_ylabel("nombre de sessions")
        ax.set_title(titre)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(G, fname), dpi=130)
        plt.close(fig)

    hist_log("duration", "1_duree.png", "Duree des sessions (secondes, echelle log)")
    hist_log("n_events", "2_n_events.png", "Nombre d'evenements par session (echelle log)")
    hist_log("orig_bytes", "3_orig_bytes.png", "Octets envoyes - HTTPS/HTTP (echelle log)", web)
    hist_log("resp_bytes", "4_resp_bytes.png", "Octets recus - HTTPS/HTTP (echelle log)", web)

    # desequilibre par dataset
    cnt = df.groupby(["dataset", "label"]).size().unstack(fill_value=0).rename(columns=nom)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cnt.plot(kind="barh", ax=ax, color=[coul[0], coul[1]], logx=True)
    ax.set_xlabel("nombre de sessions (echelle log)")
    ax.set_title("Benin / C2 par dataset")
    fig.tight_layout()
    fig.savefig(os.path.join(G, "5_desequilibre.png"), dpi=130)
    plt.close(fig)

    # taille des requetes DNS (les noms sont tronques : on prend 'size', pas len(query))
    e2 = pd.read_parquet(EVENTS, columns=["session_id", "event_type", "size"])
    e2 = e2[e2["event_type"] == "dns_query"].merge(df[["session_id", "label"]], on="session_id")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lab in (0, 1):
        v = e2.loc[e2["label"] == lab, "size"].dropna()
        if len(v):
            ax.hist(v, bins=60, alpha=0.55, color=coul[lab], density=True,
                    label=f"{nom[lab]} (n={len(v)})")
    ax.set_xlabel("taille de la requete DNS (octets)")
    ax.set_ylabel("proportion")
    ax.set_title("Taille des requetes DNS (proxy de l'entropie : les noms sont tronques)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(G, "6_taille_dns.png"), dpi=130)
    plt.close(fig)
    say(f"6 graphiques ecrits dans {G}")

with open(os.path.join(OUT, "suite_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print("\nEcrit : p3_output\\suite_report.txt, labels_a_verifier.csv, labels_summary.txt, graphs\\")
