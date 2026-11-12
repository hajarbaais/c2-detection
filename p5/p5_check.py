"""Verification rapide des sorties de p5_features.py. Usage : python p5_check.py p5_output"""
import sys
import numpy as np
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "p5_output"
t = pd.read_parquet(f"{d}/features_tls.parquet")
feats = [c for c in t.columns if c not in ["session_id", "label", "family", "session_type", "split", "unseen_strict"]]

print("== Sessions par split et label ==")
print(t.groupby(["split", "label"]).size().unstack(fill_value=0))
print("\n== NaN dans les features :", int(t[feats].isna().sum().sum()))
print("\n== Moyenne des features par label (0=benin, 1=C2) ==")
print(t.groupby("label")[feats].mean().T.round(2))

z = np.load(f"{d}/features_seq.npz")
X, m = z["X"], z["mask"]
print("\n== Sequences ==", X.shape, "| NaN :", int(np.isnan(X).sum()))
print("Pas valides (moyenne) par type de session :")
print(pd.Series(m.sum(1)).groupby(z["session_type"]).mean().round(1))
print("Sessions sans aucun pas :", int((m.sum(1) == 0).sum()))
