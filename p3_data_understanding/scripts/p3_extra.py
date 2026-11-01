r"""
P3 - Controles complementaires (a lancer apres quality.py et p3_suite.py)

Usage (depuis P2_livrables_v3\P2_livrables_v3) :
    python p3_extra.py

Fait 4 choses :
  E1. doublons de CONTENU (sessions identiques sauf leur identifiant)
  E2. fuite de machines : un meme src_ip_hash dans plusieurs jeux (train/val/test)
  E3. test de raccourci : le label se devine-t-il avec la seule source (dataset) ?
  E4. graphiques manquants : entropie des noms DNS et taille DNS par session

Sorties : p3_output\extra_report.txt, graphs\7_entropie_dns.png, graphs\8_taille_dns_session.png
"""
import math
import os
import sys
from collections import Counter

import pandas as pd

OUT = "p3_output"
UNIFIED = os.path.join(OUT, "sessions_unified.parquet")
EVENTS = os.path.join("output", "events.parquet")
if not os.path.exists(UNIFIED):
    sys.exit("Lance d'abord : python quality.py")

df = pd.read_parquet(UNIFIED)
report = []


def say(t=""):
    print(t)
    report.append(str(t))


def title(t):
    say("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# ------------------------------------------------------- E1. DOUBLONS DE CONTENU
title("E1. DOUBLONS DE CONTENU (tout sauf identifiants et horodatages)")
ignore = {"session_id", "session_key", "ts_start", "ts_end", "capture_start", "file"}
cols = [c for c in df.columns if c not in ignore and df[c].map(lambda x: not hasattr(x, "__len__") or isinstance(x, str)).all()]
dup = df.duplicated(subset=cols, keep=False)
say(f"colonnes comparees : {len(cols)}")
say(f"sessions au contenu identique a une autre : {int(dup.sum())} ({dup.mean() * 100:.1f} %)")
say("Par dataset / label :")
say(df[dup].groupby(["dataset", "label"]).size().to_string() if dup.any() else "  aucune")
if dup.any():
    # un meme contenu dans deux jeux differents = fuite possible
    g = df[dup].groupby(cols, dropna=False)["split"].nunique()
    say(f"\ngroupes de contenu identique presents dans plusieurs jeux : {int((g > 1).sum())}")
    say("Lecture : des sessions identiques sont normales pour du balisage regulier, mais si elles")
    say("sont reparties entre train et test le score de test est gonfle.")

# ------------------------------------------------------------ E2. FUITE DE MACHINES
title("E2. UNE MEME MACHINE (src_ip_hash) DANS PLUSIEURS JEUX ?")
n_split = df.groupby("src_ip_hash")["split"].nunique()
multi = n_split[n_split > 1]
say(f"machines distinctes : {len(n_split)}   presentes dans >1 jeu : {len(multi)}")
if len(multi):
    sub = df[df["src_ip_hash"].isin(multi.index)]
    say(sub.groupby(["dataset", "split"]).size().unstack(fill_value=0).to_string())
    say("-> a signaler a P2 si ce n'est pas le cas DoHBrw (decoupe par machine attendue).")
else:
    say("Aucune : decoupage par machine respecte.")

# --------------------------------------------------------- E3. TEST DE RACCOURCI
title("E3. RACCOURCI DE SOURCE : label devine avec le seul dataset")
maj = df.groupby("dataset")["label"].agg(lambda s: int(s.mean() >= 0.5))
acc = (df["dataset"].map(maj) == df["label"]).mean()
say(f"accuracy en predisant 'label majoritaire du dataset' : {acc * 100:.2f} %")
mix = df.groupby("dataset")["label"].agg(["mean", "count"]).round(4)
mix.columns = ["part_C2", "n"]
say(mix.to_string())
say("Lecture : si ce score est proche de 100 %, un modele peut reussir en reconnaissant la source")
say("(format du pcap, horodatage, port...) sans rien comprendre au C2. P6 doit evaluer aussi")
say("par dataset et ne pas se fier a un recall global.")

# --------------------------------------------------------------- E4. GRAPHIQUES
title("E4. GRAPHIQUES COMPLEMENTAIRES")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    say("matplotlib absent : python -m pip install matplotlib")

if plt:
    import numpy as np

    G = os.path.join(OUT, "graphs")
    dns_ids = df.loc[df["session_type"] == "dns", ["session_id", "label"]]
    ev = pd.read_parquet(EVENTS, columns=["session_id", "event_type", "query", "size"])
    ev = ev[ev["event_type"] == "dns_query"].merge(dns_ids, on="session_id")

    def entropie(s):
        if not isinstance(s, str) or not s:
            return float("nan")
        n = len(s)
        return -sum(k / n * math.log2(k / n) for k in Counter(s).values())

    ev["ent"] = ev["query"].map(lambda q: entropie(q.split(".")[0]) if isinstance(q, str) else float("nan"))
    ses = ev.groupby("session_id").agg(label=("label", "first"), ent=("ent", "mean"),
                                       taille=("size", "mean"), n=("size", "size")).dropna(subset=["ent"])
    say(f"{len(ses)} sessions DNS avec entropie calculable")
    say(ses.groupby("label")[["ent", "taille", "n"]].describe().round(2).T.to_string())

    coul = {0: "tab:blue", 1: "tab:red"}
    nom = {0: "benin", 1: "C2"}
    for col, fname, lab_x, titre in [
            ("ent", "7_entropie_dns.png", "entropie de Shannon du 1er label (bits), moyenne par session",
             "Entropie des noms DNS, par session (les noms sont tronques : valeur indicative)"),
            ("taille", "8_taille_dns_session.png", "taille moyenne des requetes DNS (octets), par session",
             "Taille des requetes DNS, une valeur par session")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for lab in (0, 1):
            v = ses.loc[ses["label"] == lab, col]
            if len(v):
                ax.hist(v, bins=50, alpha=0.55, color=coul[lab], density=True,
                        label=f"{nom[lab]} (n={len(v)} sessions)")
        ax.set_xlabel(lab_x)
        ax.set_ylabel("proportion")
        ax.set_title(titre)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(G, fname), dpi=130)
        plt.close(fig)
    say("2 graphiques ecrits dans " + G)

with open(os.path.join(OUT, "extra_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print("\nEcrit : p3_output\\extra_report.txt")
