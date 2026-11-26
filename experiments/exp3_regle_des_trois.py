"""Experience 3 - Regle des trois : combien de benin faut-il pour prouver une FPR < 0,1 % ?

Regle des trois : 0 fausse alerte sur N benins => FPR < 3/N (95 % de confiance). Pour FPR < f : N >= 3/f.
Compte le benin disponible par split / groupe et dit si le benin de VALIDATION suffit a calibrer 0,1 %.

Usage : python -I experiments/exp3_regle_des_trois.py --p2 <P2_output>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = 0.001


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2", required=True)
    a = ap.parse_args()

    print("=== REGLE DES TROIS : benin necessaire par FPR cible (0 FP) ===")
    for f in (0.01, 0.005, 0.001, 0.0005):
        print(f"  FPR < {f*100:>6.3f} %  ->  N >= {int(np.ceil(3/f)):>6,} benins")
    print(f"\n  objectif 0,1 % : minimum {int(np.ceil(3/TARGET)):,}, confortable ~30 000\n")

    s = pd.read_parquet(Path(a.p2) / "sessions_pcap.parquet", columns=["label", "split", "session_type"])
    s["group"] = np.where(s["session_type"] == "dns", "dns", "web")
    ben = s[s.label == 0]
    tab = ben.pivot_table(index="split", columns="group", values="label", aggfunc="size", fill_value=0)
    tab["total"] = tab.sum(axis=1)
    print("=== BENIN DISPONIBLE (par split / groupe) ===")
    print(tab.to_string())

    print("\n=== VERDICT (le seuil se regle sur le benin de VALIDATION) ===")
    val = ben[ben.split == "val"]
    for grp in ["dns", "web"]:
        n = int((val.group == grp).sum())
        ok = "OK" if n >= 3 / TARGET else "INSUFFISANT"
        print(f"  val benin {grp:4} : {n:>6,}  -> FPR prouvable < {3/max(n,1)*100:.3f} %  [{ok} pour 0,1 %]")


if __name__ == "__main__":
    main()
