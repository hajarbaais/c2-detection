r"""
P3 - Controle qualite de sessions_pcap.parquet (livre par P2)

Usage (depuis le dossier P2_livrables_v3\P2_livrables_v3) :
    python quality.py

Sorties (dossier p3_output\) :
    quality_report.txt       rapport lisible
    sessions_unified.parquet base unifiee (= sessions de P2 + 4 colonnes ajoutees)
"""
import os
import pandas as pd

IN = os.path.join("output", "sessions_pcap.parquet")
OUT_DIR = "p3_output"
os.makedirs(OUT_DIR, exist_ok=True)

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def title(t):
    say("\n" + "=" * 70)
    say(t)
    say("=" * 70)


df = pd.read_parquet(IN)
say(f"Fichier lu : {IN}  ->  {df.shape[0]} sessions, {df.shape[1]} colonnes")

# ---------------------------------------------------------------- 1. DOUBLONS
title("1. DOUBLONS")
dup_id = int(df["session_id"].duplicated().sum())
dup_row = int(df.duplicated().sum())
say(f"session_id en double      : {dup_id}   (attendu : 0)")
say(f"lignes entierement egales : {dup_row}   (attendu : 0)")

# ------------------------------------------------------- 2. COHERENCE SIMPLE
title("2. COHERENCE DES VALEURS")
say(f"label different de 0/1            : {int((~df['label'].isin([0, 1])).sum())}")
say(f"duration negative                 : {int((df['duration'] < 0).sum())}")
say(f"ts_end < ts_start                 : {int((df['ts_end'] < df['ts_start']).sum())}")
say(f"n_events < 1                      : {int((df['n_events'] < 1).sum())}")
say(f"colonnes IP en clair (src_ip/dst_ip) : {[c for c in ('src_ip', 'dst_ip') if c in df.columns]}")
unseen_benin = int(((df["split"] == "unseen") & (df["label"] == 0)).sum())
say(f"sessions benignes dans 'unseen'   : {unseen_benin}   (attendu : 0)")
multi = df.groupby("file")["split"].nunique()
say(f"pcap presents dans plusieurs jeux : {int((multi > 1).sum())}   "
    "(normal seulement pour DoHBrw malveillant, decoupe par machine)")

# --------------------------------------------------------- 3. VALEURS MANQUANTES
title("3. VALEURS MANQUANTES (par type de session)")
cols = ["server_name", "parent_domain", "orig_bytes", "resp_bytes", "ja3", "ja4",
        "ja3s", "tls_version", "cert_self_signed", "capture_start"]
miss = df.groupby("session_type")[cols].agg(lambda s: s.isna().mean() * 100).round(1)
say("Pourcentage de valeurs manquantes :")
say(miss.to_string())
say("\nLecture : orig_bytes/resp_bytes vides sur 'dns' = NORMAL (ces champs ne concernent")
say("que HTTPS/HTTP). parent_domain vide sur 'https'/'http' = NORMAL (champ DNS).")
n_cs = int(df["capture_start"].isna().sum())
say(f"capture_start vide : {n_cs} lignes -> NORMAL si 100% Stratosphere (horodatages relatifs)")
say(df[df["capture_start"].isna()]["dataset"].value_counts().to_string())

# ------------------------------------------------------------ 4. ABERRATIONS
title("4. VALEURS EXTREMES (on les signale, on ne les supprime pas)")
say(df.groupby("session_type")[["duration", "n_events"]].describe().round(1).T.to_string())
say("\nTop 10 n_events :")
say(df.nlargest(10, "n_events")[["dataset", "family", "session_type", "n_events", "duration"]].to_string())
say("\nTop 10 duration :")
say(df.nlargest(10, "duration")[["dataset", "family", "session_type", "n_events", "duration"]].to_string())

# --------------------------------------------------- 5. DESEQUILIBRE ET VOLUME
title("5. DESEQUILIBRE C2 / BENIN ET VOLUME")
n_c2 = int(df["label"].sum())
say(f"Total : {len(df) - n_c2} benins / {n_c2} C2  ({n_c2 / len(df) * 100:.2f} % de C2)")
say("\nPar jeu :")
say(df.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns={0: "benin", 1: "c2"}).to_string())
say("\nPar dataset :")
say(df.groupby(["dataset", "label"]).size().unstack(fill_value=0).rename(columns={0: "benin", 1: "c2"}).to_string())
say("\nPar type de session :")
say(df.groupby(["session_type", "label"]).size().unstack(fill_value=0).rename(columns={0: "benin", 1: "c2"}).to_string())
benin_test = int(((df["split"] == "test") & (df["label"] == 0)).sum())
ok = "OK" if benin_test >= 100000 else "ALERTE : sous le seuil de la fiche"
say(f"\nBenins dans 'test' : {benin_test}  (seuil 100 000)  -> {ok}")
say("\nJeux petits (metriques instables) :")
for s in ("val", "unseen"):
    say(f"  {s} : {int(((df['split'] == s) & (df['label'] == 1)).sum())} sessions C2")

# ------------------------------------------------ 6. COLONNES AJOUTEES PAR P3
title("6. COLONNES AJOUTEES PAR P3")
df["source"] = "pcap"

web = df["session_type"].isin(["https", "http"])
# 'pas de TLS vu' : session HTTPS sans empreinte JA3 (pas de handshake dans la capture)
df["no_tls_seen"] = (df["session_type"] == "https") & df["ja3"].isna()
say(f"no_tls_seen = True : {int(df['no_tls_seen'].sum())} sessions HTTPS sans JA3")
say(df[df["no_tls_seen"]].groupby(["dataset", "label"]).size().to_string())

# part de connexions sans charge utile (scans CTU-13...) - seulement HTTPS/HTTP
share = (df["n_conn_no_payload"] / df["n_events"]).where(web)
df["share_no_payload"] = share.clip(upper=1.0)
say(f"\nshare_no_payload moyen (HTTPS/HTTP) par label :")
say(df[web].groupby("label")["share_no_payload"].mean().round(3).to_string())

# CTU-13 : toute session de la machine infectee est C2, meme le DNS banal
df["ctu13_broad_label"] = (df["dataset"] == "ctu13") & (df["label"] == 1) & (df["session_type"] == "dns")
say(f"\nctu13_broad_label = True : {int(df['ctu13_broad_label'].sum())} sessions DNS CTU-13 etiquetees C2")
say("-> permet a P6 de tester AVEC et SANS ces sessions (filtre : ~ctu13_broad_label).")

# aberrations : simple drapeau, p99.9 par type de session
q = df.groupby("session_type")["n_events"].transform(lambda s: s.quantile(0.999))
df["is_extreme_events"] = df["n_events"] > q
say(f"\nis_extreme_events = True : {int(df['is_extreme_events'].sum())} sessions (au-dessus du p99.9 de leur type)")

# --------------------------------------------------- 7. PIEGES CONNUS (rappel)
title("7. PIEGES A SIGNALER A L'EQUIPE")
t = df[(df["dataset"] == "stratosphere-https-c2") & (df["n_events"] == 6)]
say(f"- TrickBot/Dridex a 6 evenements : {len(t)} sessions tres repetitives, "
    f"reparties sur {t['file'].nunique()} pcap -> risque de gonfler les scores.")
say(f"- DNS-EXF : {int(((df['dataset'] == 'cic-bell-dns-exf-2021') & (df['label'] == 1)).sum())} sessions C2 "
    f"seulement contre {int(((df['dataset'] == 'cic-bell-dns-exf-2021') & (df['label'] == 0)).sum())} benignes.")
say(f"- Stratosphere avec tls_intercepted=True : {int((df['tls_intercepted'] == True).sum())} sessions "
    "(ignorer certificat / auto-signe / JA3S).")
say("- Colonnes INTERDITES comme features : src_ip_hash, dst_ip_hash, session_key, "
    "qname_truncated_share (et qname_truncated dans events).")

# ------------------------------------------------------------------ SORTIES
out_pq = os.path.join(OUT_DIR, "sessions_unified.parquet")
df.to_parquet(out_pq, index=False)
with open(os.path.join(OUT_DIR, "quality_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
say(f"\nEcrit : {out_pq}  ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
say(f"Ecrit : {os.path.join(OUT_DIR, 'quality_report.txt')}")
