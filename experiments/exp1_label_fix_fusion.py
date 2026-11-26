"""Experience 1 - Fusion de features + correction des labels (DNS + web, par session).

Un seul modele supervise nourri avec TOUTES les features (P4 entropie/temps + P5 TLS + resume des
sequences de paquets), apres avoir retire les C2 'convention' de CTU-13 (drapeau ctu13_broad_label de P3).
Seuil calibre a 0,1 % FPR sur un grand ensemble benin (moitie du benin de test), FPR mesuree sur l'autre moitie.

Resultat attendu : DNS ~100 % de recall a <0,1 % FPR ; web ~10 % (dridex ~40 %) a 0,1 % FPR.

Usage :
  python -I experiments/exp1_label_fix_fusion.py --p2 <P2_output> --unified <.../sessions_unified.parquet>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_exp import (RANDOM_STATE, add_common_args, feature_columns, load_all, metrics_by,
                        threshold_at_fpr, to_matrix, train_mlp, wilson_ci)


def run_group(df, group, a):
    seq_cols = [c for c in df.columns if c.startswith("seq_")]
    static = feature_columns(df, group, with_p5=True)
    g = df[df.group == group].copy().reset_index(drop=True)
    Xs = to_matrix(g, static).astype("float32")
    Xq = np.nan_to_num(g[seq_cols].to_numpy("float32"), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.concatenate([Xs, Xq], axis=1)
    y = g.label.to_numpy().astype("float32")
    sp = g.split.to_numpy()
    tr, va, te = sp == "train", sp == "val", sp == "test"
    s = train_mlp(X, y, tr, va)

    teb = np.flatnonzero(te & (y == 0))
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(teb)
    h = len(teb) // 2
    calib = np.concatenate([np.flatnonzero(va & (y == 0)), teb[:h]])
    evb = teb[h:]
    c2 = np.flatnonzero(te & (y == 1))
    thr = threshold_at_fpr(s[calib], 0.001)
    tp, fp = int((s[c2] > thr).sum()), int((s[evb] > thr).sum())
    rlo, rhi = wilson_ci(tp, len(c2))
    print(f"\n=== {group} : recall {tp/len(c2)*100:.1f}% (IC {rlo*100:.0f}-{rhi*100:.0f})  "
          f"FPR {fp/len(evb)*100:.3f}%  (test {len(c2)} C2 / {len(evb)} benin)")
    fam = g.loc[c2].assign(label=1, pred=s[c2] > thr)
    ff = metrics_by(fam, "family")[["family", "n_c2", "recall"]].sort_values("recall", ascending=False)
    for _, r in ff.iterrows():
        print(f"    {r.family:22} n={int(r.n_c2):>4}  recall={r.recall*100:5.1f}%")


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--groups", default="dns,web")
    a = ap.parse_args()
    print("[*] chargement features + correction des labels (ctu13_broad_label)...")
    df = load_all(a, with_seq=True)
    for grp in [x.strip() for x in a.groups.split(",") if x.strip()]:
        run_group(df, grp, a)


if __name__ == "__main__":
    main()
