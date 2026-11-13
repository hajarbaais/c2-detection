"""P5.6b - Entraînement complet de l'autoencodeur sur le trafic BÉNIN, suivi dans MLflow.

Réutilise le prototype P5.6a (prétraitement, architecture, boucle d'entraînement) et ajoute :
  - entraînement sur TOUT le bénin du train (--sample-frac 1 par défaut) ;
  - ja3_rarity, ja4_rarity et log_domain_reputation (P5) RECALCULÉS sur le bénin du train seul
    (dans P5 ils étaient calculés sur tout le train, C2 compris) ;
  - suivi MLflow : paramètres, courbes de perte par époque, métriques val/test/unseen, modèle ;
  - sauvegarde de p6_models/models/autoencoder.pt (les deux groupes dns + web dans un seul fichier) ;
  - load_autoencoder() / score_autoencoder() pour brancher le modèle dans le pipeline P7.

Un modèle par groupe (dns / web), comme P6. Seuil : bénin de validation, 0,1 % de FPR.

Sorties : p6_models/models/autoencoder.pt          (gitignoré : ajouter `p6_models/models/*.pt`)
          p6_models/evaluation/autoencoder_metrics.csv      recall / FPR / AUC par groupe et split
          p6_models/evaluation/autoencoder_by_family.csv    par famille (test)

Usage : python -I p6_models/scripts/p5_6b_autoencoder_train.py --p2 <dossier avec sessions_pcap.parquet>
            --p5 <dossier p5_output> [--p3 <sessions_unified.parquet>] [--epochs 100] [--no-mlflow]
Puis  : mlflow ui --backend-store-uri sqlite:///mlflow.db   (depuis la racine du dépôt)
"""
import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from p5_6a_autoencoder_proto import (IFOREST_AUC, P5_COLS, P5_FEATS, AutoEncoder, Prep,  # noqa: E402
                                     recon_error, train)
from p6_common import (EVAL_DIR, GROUPS, MODELS_DIR, RANDOM_STATE, SESSION_NUMERIC, TARGET_FPR,  # noqa: E402
                       feature_columns, git_commit, load_dataset, p2_dir, setup_mlflow)
from metrics import binary_metrics, metrics_by, threshold_at_fpr  # noqa: E402


# --------------------------------------------------------------------------- features P5 sur le bénin seul
def _rarity(col: pd.Series, ref: pd.Series) -> pd.Series:
    """-log(fréquence dans la référence). Valeur jamais vue -> rareté maximale. Pas de JA3/JA4 -> 0."""
    freq = col[ref].value_counts(normalize=True)
    max_rar = -np.log(1.0 / (ref.sum() + 1))
    rar = (-np.log(col.map(freq).astype("float64"))).fillna(max_rar)
    rar[col.isna()] = 0.0
    return rar


def recompute_p5_on_benign(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcule rareté JA3/JA4 et réputation du domaine en n'utilisant que le BÉNIN du train web."""
    ref = (df["split"] == "train") & (df["label"] == 0) & (df["group"] == "web")
    df = df.copy()
    df["p5_ja3_rarity"] = _rarity(df["ja3"], ref)
    df["p5_ja4_rarity"] = _rarity(df["ja4"], ref)
    key = df["server_name"].fillna(df["dst_ip_hash"])
    rep = df.loc[ref].groupby(key[ref])["src_ip_hash"].nunique()
    df["p5_log_domain_reputation"] = np.log1p(key.map(rep).fillna(0))
    return df


# --------------------------------------------------------------------------- expérience : re-split du bénin web
def resplit_benign_web(df: pd.DataFrame, gap: float = 0.02) -> pd.DataFrame:
    """EXPÉRIENCE (ne remplace pas le découpage de P2) : tester si le déséquilibre du bénin web explique le FPR.

    Le bénin web, aujourd'hui presque entièrement dans le test (87,5 %), est redécoupé DANS CHAQUE (PCAP, FAMILLE)
    selon l'ordre des horodatages (le bénin DoH est groupé à la fin de 3 gros pcaps Firefox : sans la famille dans
    la clé, il resterait entièrement dans le test) : 60 % début -> train, 15 % -> val, 21 % fin -> test, avec un écart de 2 % (sessions écartées)
    entre deux ensembles pour éviter les fenêtres à cheval. Les C2 et le DNS ne sont pas touchés.
    ATTENTION : mêmes machines et sessions voisines dans le temps entre train et test -> le résultat est une
    BORNE OPTIMISTE de l'amélioration possible, pas une mesure de généralisation.
    """
    df = df.copy()
    m = (df["group"] == "web") & (df["label"] == 0) & (df["split"] == "test")
    rank = df.loc[m].groupby(["file", "family"])["ts_start"].rank(method="first", pct=True)
    new = pd.Series("test", index=rank.index)
    new[rank <= 0.60] = "train"
    new[(rank > 0.60) & (rank <= 0.60 + gap)] = "gap"
    new[(rank > 0.60 + gap) & (rank <= 0.75 + gap)] = "val"
    new[(rank > 0.75 + gap) & (rank <= 0.75 + 2 * gap)] = "gap"
    df.loc[m, "split"] = new
    return df


# --------------------------------------------------------------------------- rechargement (pipeline P7)
def prep_from_dict(d: dict) -> Prep:
    p = Prep()
    p.cols, p.log_cols, p.miss_cols = d["cols"], d["log_cols"], d["miss_cols"]
    p.median, p.mean, p.std = (pd.Series(d[k]).reindex(p.cols) for k in ("median", "mean", "std"))
    return p


def load_autoencoder(path) -> dict:
    """Recharge autoencoder.pt -> {groupe: {model, prep, threshold, cols}} (une seule fois au démarrage)."""
    ckpt = torch.load(path, weights_only=False)
    out = {}
    for group, b in ckpt["groups"].items():
        model = AutoEncoder(**b["dims"])
        model.load_state_dict(b["state_dict"])
        model.eval()
        out[group] = {"model": model, "prep": prep_from_dict(b["prep"]),
                      "threshold": b["threshold"], "cols": b["prep"]["cols"]}
    return out


def score_autoencoder(loaded: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Score (erreur de reconstruction) et alerte pour chaque session de df (colonne `group` requise)."""
    out = pd.DataFrame({"ae_score": np.nan, "ae_alert": False}, index=df.index)
    for group, b in loaded.items():
        m = (df["group"] == group).to_numpy()
        if m.any():
            s = recon_error(b["model"], b["prep"].transform(df[m]))
            out.loc[m, "ae_score"] = s
            out.loc[m, "ae_alert"] = s > b["threshold"]
    return out


# --------------------------------------------------------------------------- un groupe
def run_group(df, group, use_p5, a, mlflow):
    g = df[df["group"] == group].reset_index(drop=True)
    cols = feature_columns(df, group)
    if group == "web" and use_p5:
        cols = [c for c in cols if c not in SESSION_NUMERIC] + P5_FEATS
    tr = g[(g["split"] == "train") & (g["label"] == 0)]
    if a.sample_frac < 1:
        tr = tr.sample(frac=a.sample_frac, random_state=RANDOM_STATE)
    va = g[(g["split"] == "val") & (g["label"] == 0)]
    print(f"\n=== {group} : {len(cols)} colonnes, train bénin {len(tr):,}, val bénin {len(va):,}")

    prep = Prep().fit(tr[cols])
    Xtr, Xva = prep.transform(tr[cols]), prep.transform(va[cols])
    model = AutoEncoder(prep.n_out, a.h1, a.h2, a.z)
    t0 = time.perf_counter()
    hist = train(model, Xtr, Xva, a.epochs, a.lr, a.batch, a.patience)
    fit_s = time.perf_counter() - t0
    best = min(hist, key=lambda h: h[2])
    print(f"    {len(hist)} époques en {fit_s:.1f} s ; meilleure perte val {best[2]:.4f} (époque {best[0]})")

    thr = threshold_at_fpr(recon_error(model, Xva), TARGET_FPR)
    if len(va) * TARGET_FPR < 5:
        print(f"    ! {len(va) * TARGET_FPR:.1f} faux positifs autorisés en val : seuil peu fiable")

    rows, fam = [], None
    for split in ["train", "val", "test", "unseen"]:
        s = g[g["split"] == split]
        sc = recon_error(model, prep.transform(s[cols]))
        r = binary_metrics(s["label"], sc > thr, sc)
        rows.append({"group": group, "split": split, **r})
        if group == "dns" and split in ("val", "test") and "ctu13_broad_label" in s:
            # sans les requêtes DNS banales de CTU-13 (C2 « par convention », P3) : même seuil
            keep = ~s["ctu13_broad_label"].fillna(False).astype(bool).to_numpy()
            rows.append({"group": group, "split": f"{split}_sans_ctu13_broad",
                         **binary_metrics(s["label"].to_numpy()[keep], (sc > thr)[keep], sc[keep])})
        if split == "test":
            fam = metrics_by(s.assign(pred=sc > thr), "family").assign(group=group)
    strict = g[(g["split"] == "unseen") & (g["unseen_strict"] == True)]  # noqa: E712
    if len(strict):
        rows.append({"group": group, "split": "unseen_strict",
                     **binary_metrics(strict["label"], recon_error(model, prep.transform(strict[cols])) > thr)})

    if mlflow:
        mlflow.log_params({f"{group}_n_train": len(tr), f"{group}_n_features": len(cols),
                           f"{group}_input_dim": prep.n_out, f"{group}_epochs_run": len(hist)})
        mlflow.log_metric(f"{group}/threshold", float(thr))
        mlflow.log_metric(f"{group}/fit_seconds", fit_s)
        for ep, tl, vl in hist:
            mlflow.log_metric(f"train/{group}/loss", tl, step=ep)
            mlflow.log_metric(f"val/{group}/loss", vl, step=ep)
        for r in rows:
            for k in ("recall", "fpr", "precision", "f1", "roc_auc", "pr_auc"):
                v = r.get(k)
                if v is not None and not np.isnan(v):
                    mlflow.log_metric(f"{r['split']}/{group}/{k}", float(v))
        mlflow.log_text("\n".join(cols), f"features_{group}.txt")

    bundle = {"state_dict": model.state_dict(), "dims": model.dims, "prep": prep.to_dict(),
              "threshold": float(thr), "history": hist}
    return bundle, rows, fam


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2", help="dossier contenant sessions_pcap.parquet")
    ap.add_argument("--p5", help="dossier p5_output : ajoute les features TLS au groupe web")
    ap.add_argument("--p3", help="sessions_unified.parquet (P3) : ajoute ctu13_broad_label pour l'ÉVALUATION "
                                 "(jamais utilisée comme feature)")
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--h1", type=int, default=64)
    ap.add_argument("--h2", type=int, default=32)
    ap.add_argument("--z", type=int, default=8)
    ap.add_argument("--run-name", default="ae_v1")
    ap.add_argument("--resplit-benign-web", action="store_true",
                    help="EXPÉRIENCE : redécoupe le bénin web par pcap (60/15/21 %%) ; sorties suffixées _resplit")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--p5-global-rarity", action="store_true",
                    help="garder la rareté P5 d'origine (calculée sur tout le train) au lieu du bénin seul")
    a = ap.parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    df = load_dataset(p2_dir(a.p2)).reset_index(drop=True)
    if a.resplit_benign_web:
        df = resplit_benign_web(df)
        print("[expérience] bénin web redécoupé :",
              df[(df["group"] == "web") & (df["label"] == 0)]["split"].value_counts().to_dict())
    suffix = "_resplit" if a.resplit_benign_web else ""
    if a.p5:
        t = pd.read_parquet(Path(a.p5) / "features_tls.parquet")[["session_id"] + P5_COLS]
        df = df.merge(t.rename(columns={c: "p5_" + c for c in P5_COLS}), on="session_id", how="left",
                      validate="one_to_one")
        if not a.p5_global_rarity:
            df = recompute_p5_on_benign(df)

    if a.p3:
        u = pd.read_parquet(a.p3, columns=["session_id", "ctu13_broad_label"])
        df = df.merge(u, on="session_id", how="left", validate="one_to_one")

    mlflow = None if a.no_mlflow else setup_mlflow("c2-autoencoder")
    run = mlflow.start_run(run_name=a.run_name) if mlflow else nullcontext()
    bundles, all_rows, fams = {}, [], []
    with run:
        if mlflow:
            mlflow.set_tags({"lot": "P5", "task": "P5.6", "author": "oumaima", "git_commit": git_commit(),
                             "experiment": "resplit_benign_web" if a.resplit_benign_web else "split_P2", "features": "p4_features_dns" + (" + p5_tls (rareté sur bénin)" if a.p5 else "")})
            mlflow.log_params({"h1": a.h1, "h2": a.h2, "z": a.z, "lr": a.lr, "batch": a.batch,
                               "epochs_max": a.epochs, "patience": a.patience, "sample_frac": a.sample_frac,
                               "target_fpr": TARGET_FPR, "threshold_split": "val_benign", "seed": RANDOM_STATE})
        for group in GROUPS:
            b, rows, fam = run_group(df, group, bool(a.p5), a, mlflow)
            bundles[group] = b
            all_rows += rows
            fams.append(fam)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        pt = MODELS_DIR / f"autoencoder{suffix}.pt"
        torch.save({"groups": bundles, "seed": RANDOM_STATE, "target_fpr": TARGET_FPR}, pt)
        res = pd.concat([pd.DataFrame(all_rows)], ignore_index=True)
        res.to_csv(EVAL_DIR / f"autoencoder_metrics{suffix}.csv", index=False, float_format="%.6f")
        pd.concat(fams).to_csv(EVAL_DIR / f"autoencoder_by_family{suffix}.csv", index=False, float_format="%.6f")
        if mlflow:
            mlflow.log_artifact(str(pt), artifact_path="models")
            mlflow.log_artifact(str(EVAL_DIR / f"autoencoder_metrics{suffix}.csv"))

    print(f"\nModèle : {pt}")
    view = res[res.split.isin(["val", "test", "val_sans_ctu13_broad", "test_sans_ctu13_broad", "unseen", "unseen_strict"])][
        ["group", "split", "n_c2", "n_benign", "recall", "fpr", "roc_auc"]].copy()
    view["iforest_auc"] = [IFOREST_AUC.get(g) if s == "test" else None for g, s in zip(view.group, view.split)]
    print(view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nObjectifs du prof : recall >= 0,98 et FPR < 0,001 (à mesurer sur le test, jamais à annoncer avant).")


if __name__ == "__main__":
    main()
