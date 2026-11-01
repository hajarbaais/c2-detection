"""P2 - tâche 7 : découpage train / val / test anti-fuite + jeu « familles non vues ».

Règles (fiche P2 + docs/P2_HANDOFF.md § 6) :
  1. Unité de découpage = un GROUPE, jamais une session isolée :
       - DoHBrw malveillant : la machine (doh1 ... doh10, dans le nom du fichier) -> anti-fuite « machine »
       - toutes les autres sources : le pcap (= une machine sur une période)
     => deux sessions d'un même pcap (ou d'une même machine DoHBrw) ne sont jamais dans deux jeux.
  2. Dans chaque strate (dataset x classe), les groupes sont triés par date de capture : les plus anciens
     vont en train, puis val, puis test (~70 / 15 / 15 % des sessions) -> anti-fuite « période ».
  3. Familles non vues : dnscat2, murlo, nsis.ay -> jeu « unseen » (aucune session en train/val/test).
  4. cobaltstrike (250 sessions seulement) -> test (conseil de P1).
  5. firefox-adguard et firefox-cloudflare (DoHBrw bénin, 1 seule capture de 5 Go chacune) -> test, pour
     atteindre ~100 000 sessions bénignes en test (FPR < 0,1 %, accord avec P6) ; firefox-quad9 reste en train.

Sorties (p2/output/splits/) : {train,val,test,unseen}_ids.csv, split_assignment.csv (groupe -> jeu),
split_summary.csv (comptes), et une colonne `split` ajoutée à sessions_pcap.parquet.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from p2_common import OUT, SPLITS

UNSEEN_FAMILIES = {"dnscat2", "murlo", "nsis.ay"}
FORCE_TEST_FAMILIES = {"cobaltstrike"}
FORCE_TEST_FILES = {
    "cira-cic-dohbrw-2020/pcap/benign/firefox-adguard/dump.pcap",
    "cira-cic-dohbrw-2020/pcap/benign/firefox-cloudflare/dump.pcap",
}
FRACTIONS = (0.70, 0.85)          # bornes cumulées train | val | test


def split_group(row):
    if row["dataset"] == "cira-cic-dohbrw-2020" and row["label"] == 1:
        m = re.search(r"_doh(\d+)_", row["file"])
        return f"dohbrw-machine-doh{m.group(1)}"
    return row["file"]


def stratum(row):
    """Sous-population dans laquelle on applique l'ordre chronologique 70/15/15, pour que chaque variante
    (intensité d'exfiltration, couple navigateur-résolveur, famille de malware) soit présente dans chaque jeu
    dès qu'elle a au moins 3 captures."""
    ds = row["dataset"]
    if ds == "cic-bell-dns-exf-2021":
        return row["capture_group"]                       # benign | heavy/attack | light/attack | ...
    if ds == "cira-cic-dohbrw-2020":
        return "machines" if row["label"] == 1 else row["capture_group"]   # benign/chrome-adguard ...
    if ds in ("ctu13", "stratosphere-https-c2"):
        return row["family"]
    return "all"


def assign(groups):
    """groups : DataFrame (group, start, n) d'une strate -> Series group -> split."""
    g = groups.sort_values(["start", "group"], na_position="last").reset_index(drop=True)
    total = g["n"].sum()
    mid = (g["n"].cumsum() - g["n"] / 2) / total
    split = pd.Series(["train"] * len(g))
    split[mid >= FRACTIONS[0]] = "val"
    split[mid >= FRACTIONS[1]] = "test"
    if len(g) >= 3:                       # val et test jamais vides quand c'est possible
        if (split == "test").sum() == 0:
            split.iloc[-1] = "test"
        if (split == "val").sum() == 0:
            split.iloc[-2] = "val"
    elif len(g) == 2:
        split.iloc[-1] = "test"
    return pd.Series(split.values, index=g["group"])


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    S = pd.read_parquet(OUT / "sessions_pcap.parquet")
    S["split_group"] = S.apply(split_group, axis=1)
    S["split"] = None
    S.loc[S["family"].isin(UNSEEN_FAMILIES), "split"] = "unseen"
    S.loc[S["family"].isin(FORCE_TEST_FAMILIES), "split"] = "test"
    S.loc[S["file"].isin(FORCE_TEST_FILES), "split"] = "test"

    S["stratum"] = S.apply(stratum, axis=1)
    rest = S[S["split"].isna()]
    for _, part in rest.groupby(["dataset", "label", "stratum"]):
        groups = part.groupby("split_group").agg(start=("capture_start", "min"), n=("session_id", "size")) \
                     .reset_index().rename(columns={"split_group": "group"})
        mapping = assign(groups)
        S.loc[part.index, "split"] = part["split_group"].map(mapping)

    # Les 10 machines DoHBrw ont toutes lancé les 3 outils : dnscat2 (non vu) partage donc ses machines avec
    # train. unseen_strict = sessions non vues enregistrées sur une machine du jeu test (ni la famille ni la
    # machine n'ont été vues à l'entraînement) -> mesure de généralisation la plus stricte.
    machine_split = S[S["split"] != "unseen"].groupby("split_group")["split"].first()
    S["unseen_strict"] = (S["split"] == "unseen") & (
        S["split_group"].map(machine_split).isna() | (S["split_group"].map(machine_split) == "test"))

    # contrôles anti-fuite (entre train / val / test ; unseen est un jeu d'évaluation à part)
    core = S[S["split"] != "unseen"]
    assert (core.groupby("split_group")["split"].nunique() == 1).all(), "un groupe est dans plusieurs jeux"
    assert (S.groupby("file")["split"].nunique() == 1).all(), "un pcap est dans plusieurs jeux"

    assignment = S.groupby(["dataset", "split_group", "split"]).agg(
        families=("family", lambda f: ",".join(sorted(set(f)))),
        capture_start=("capture_start", "min"), n_sessions=("session_id", "size"),
        n_c2=("label", "sum"), n_files=("file", "nunique")).reset_index()
    assignment.to_csv(SPLITS / "split_assignment.csv", index=False)

    summary = S.groupby(["split", "dataset", "label"]).size().unstack("label", fill_value=0) \
               .rename(columns={0: "benign", 1: "c2"}).reset_index()
    summary.to_csv(SPLITS / "split_summary.csv", index=False)
    for sp in ["train", "val", "test", "unseen"]:
        S.loc[S["split"] == sp, ["session_id"]].to_csv(SPLITS / f"{sp}_ids.csv", index=False)
    S.drop(columns=["split_group", "stratum"]).to_parquet(OUT / "sessions_pcap.parquet", index=False)

    tot = S.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns={0: "benign", 1: "c2"})
    print(tot)
    print("\nfamilles C2 par jeu :")
    print(S[S.label == 1].groupby("split")["family"].agg(lambda f: sorted(set(f))))


if __name__ == "__main__":
    main()
