"""Experience 2 - Detection au niveau PAIRE (machine <-> serveur) pour le C2 chiffre (web).

Au lieu de classer chaque session, on classe la paire (src_ip_hash, dst_ip_hash) : un C2 = une paire
qui balise de facon persistante. Indicateurs de beaconing par paire (regularite des intervalles et
des tailles, periodicite, persistance). Seuil calibre sur un grand ensemble benin (moitie du benin test).

Resultat attendu : web ~63 % de recall a 0,5 % FPR (trickbot/dridex/dns tunnels detectes) ;
~13 % a 0,1 % FPR (mur du chiffre).

Usage :
  python -I experiments/exp2_pair_beaconing.py --p2 <P2_output> --unified <.../sessions_unified.parquet>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_exp import (REPO, RANDOM_STATE, add_common_args, load_all, metrics_by,
                        threshold_at_fpr, train_mlp, wilson_ci)

sys.path.insert(0, str(REPO / "p4_features_dns_time" / "scripts"))
from p4_04_periodicity import periodicity  # noqa: E402

MINC = 3   # une paire doit avoir >= MINC connexions pour parler de beaconing
FEATS = ["n_conn", "iat_cv", "iat_median", "beacon_score", "size_cv", "size_mean", "dur_cv", "span_h"]


def build_pairs(web):
    rows = []
    for _, g in web.groupby(["src_ip_hash", "dst_ip_hash"]):
        if len(g) < MINC:
            continue
        ts = np.sort(g["ts_start"].to_numpy(float))
        dt = np.diff(ts); dt = dt[dt >= 0]
        ob = g["orig_bytes"].to_numpy(float)
        dur = g["duration"].to_numpy(float)
        _, beac = periodicity(ts) if len(ts) >= 10 else (np.nan, 0.0)
        rows.append(dict(
            n_conn=len(g),
            iat_cv=float(dt.std() / dt.mean()) if len(dt) and dt.mean() > 0 else 0.0,
            iat_median=float(np.median(dt)) if len(dt) else 0.0,
            beacon_score=beac,
            size_cv=float(ob.std() / ob.mean()) if ob.mean() > 0 else 0.0,
            size_mean=float(np.log1p(ob.mean())),
            dur_cv=float(dur.std() / dur.mean()) if dur.mean() > 0 else 0.0,
            span_h=float((ts[-1] - ts[0]) / 3600.0),
            label=int(g["label"].max()),
            family=g["family"].mode().iloc[0] if g["label"].max() else "benign",
            split=g["split"].mode().iloc[0]))
    return pd.DataFrame(rows)


def main():
    a = add_common_args(argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)).parse_args()
    print("[*] chargement + correction des labels ...")
    df = load_all(a, with_seq=False)
    web = df[df.group == "web"].copy()
    print("[*] agregation par paire (src <-> dst) ...")
    P = build_pairs(web)
    print(f"    {len(P):,} paires (>= {MINC} conn) : {int(P.label.sum())} C2 / {int((P.label==0).sum())} benin")

    X = np.nan_to_num(P[FEATS].to_numpy("float32"), nan=0.0, posinf=0.0, neginf=0.0)
    y = P.label.to_numpy().astype("float32")
    sp = P.split.to_numpy()
    tr, va, te = sp == "train", sp == "val", sp == "test"
    s = train_mlp(X, y, tr, va, hidden=(64,), epochs=200, patience=15, batch=256)

    teb = np.flatnonzero(te & (y == 0))
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(teb)
    h = len(teb) // 2
    calib = np.concatenate([np.flatnonzero(va & (y == 0)), teb[:h]])
    evb = teb[h:]
    c2 = np.flatnonzero(te & (y == 1))
    print(f"\npaires test : {len(c2)} C2 / {len(evb)} benin held-out (calib {len(calib)})")
    for target in (0.001, 0.005):
        thr = threshold_at_fpr(s[calib], target)
        tp, fp = int((s[c2] > thr).sum()), int((s[evb] > thr).sum())
        rlo, rhi = wilson_ci(tp, len(c2))
        print(f"\n[@ FPR {target*100:.1f}%] recall {tp/len(c2)*100:.1f}% (IC {rlo*100:.0f}-{rhi*100:.0f})  "
              f"FPR reelle {fp/len(evb)*100:.3f}%")
        if target == 0.005:
            fam = P.loc[c2].assign(pred=s[c2] > thr)
            for f, gg in fam.groupby("family"):
                print(f"    {f:14} n={len(gg):>3} recall={gg.pred.mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
