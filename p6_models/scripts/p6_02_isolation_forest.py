"""P6.2 - Isolation Forest : détecteur d'anomalies classique (référence non supervisée).

Principe : on apprend UNIQUEMENT le trafic bénin du train. Une session « facile à isoler »
(peu de coupes aléatoires suffisent à la séparer des autres) est anormale -> score élevé.

Un modèle par groupe de trafic (features différentes) :
  dns : features de contenu DNS (entropie, forme, NXDOMAIN...) + rythme + fenêtres longues
  web : rythme (IAT, périodicité, fenêtres) + volumes bruts de P2 (en attendant les features P5)

Seuil : choisi sur le BÉNIN DE VALIDATION pour 0,1 % de FPR (le test n'est jamais regardé).
Évaluation : val, test, et « unseen » (familles jamais vues : recall seulement).

Option --supervised : ajoute un modèle supervisé de référence (HistGradientBoosting, entraîné
sur train avec labels) pour mesurer l'écart entre non supervisé et supervisé.

Sorties :
  p6_models/models/iforest_<groupe>.joblib        modèle + colonnes + seuil (gitignoré)
  p6_models/evaluation/iforest_metrics.csv        métriques par groupe et split
  p6_models/evaluation/iforest_by_family.csv      recall / FPR par famille (test)
  p6_models/output/iforest_scores.parquet         score par session (gitignoré)

Usage : python -I p6_02_isolation_forest.py [--p2 DOSSIER_OUTPUT_P2] [--supervised] [--no-mlflow]
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from p6_common import (EVAL_DIR, GROUPS, MODELS_DIR, RANDOM_STATE, REPO, TARGET_FPR,  # noqa: E402
                       feature_columns, git_commit, load_dataset, p2_dir, setup_mlflow, to_matrix)
from metrics import binary_metrics, metrics_by, threshold_at_fpr  # noqa: E402

IF_PARAMS = dict(n_estimators=300, max_samples=256, contamination="auto", random_state=RANDOM_STATE, n_jobs=-1)
HGB_PARAMS = dict(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=RANDOM_STATE)


def train_iforest(df: pd.DataFrame, group: str) -> dict:
    """Entraîne l'IF du groupe sur le bénin du train, calibre le seuil sur le bénin de val."""
    cols = feature_columns(df, group)
    g = df[df["group"] == group]
    train_benign = g[(g["split"] == "train") & (g["label"] == 0)]
    val_benign = g[(g["split"] == "val") & (g["label"] == 0)]
    t0 = time.perf_counter()
    model = IsolationForest(**IF_PARAMS).fit(to_matrix(train_benign, cols))
    fit_s = time.perf_counter() - t0
    # sklearn : score_samples élevé = normal -> on inverse pour avoir « élevé = suspect »
    val_scores = -model.score_samples(to_matrix(val_benign, cols))
    thr = threshold_at_fpr(val_scores, TARGET_FPR)
    return {"model": model, "columns": cols, "threshold": thr, "group": group,
            "n_train": len(train_benign), "fit_seconds": fit_s}


def score(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    return -bundle["model"].score_samples(to_matrix(df, bundle["columns"]))


def train_supervised(df: pd.DataFrame, group: str) -> dict:
    cols = feature_columns(df, group)
    g = df[df["group"] == group]
    tr, va = g[g["split"] == "train"], g[(g["split"] == "val") & (g["label"] == 0)]
    model = HistGradientBoostingClassifier(**HGB_PARAMS).fit(tr[cols].astype("float64"), tr["label"])
    thr = threshold_at_fpr(model.predict_proba(va[cols].astype("float64"))[:, 1], TARGET_FPR)
    return {"model": model, "columns": cols, "threshold": thr, "group": group, "supervised": True}


def score_supervised(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    return bundle["model"].predict_proba(df[bundle["columns"]].astype("float64"))[:, 1]


def evaluate(df, scores, preds, name) -> list[dict]:
    rows = []
    for split in ["train", "val", "test", "unseen"]:
        for grp in ["all"] + list(GROUPS):
            m = (df["split"] == split) & ((df["group"] == grp) if grp != "all" else True)
            if m.any():
                sc = scores[m] if grp != "all" else None   # scores de 2 modèles : pas comparables entre eux
                rows.append({"model": name, "split": split, "group": grp,
                             **binary_metrics(df.loc[m, "label"], preds[m], sc)})
    return rows


def run(df, kind, mlflow):
    fit, sc_fn = (train_iforest, score) if kind == "iforest" else (train_supervised, score_supervised)
    scores = np.full(len(df), np.nan)
    preds = np.zeros(len(df), dtype=bool)
    bundles = {}
    for group in GROUPS:
        b = fit(df, group)
        m = (df["group"] == group).to_numpy()
        scores[m] = sc_fn(b, df[m])
        preds[m] = scores[m] > b["threshold"]
        bundles[group] = b
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(b, MODELS_DIR / f"{kind}_{group}.joblib")
        print(f"       {kind}/{group}: {len(b['columns'])} features, seuil={b['threshold']:.4f}")
    rows = evaluate(df, scores, preds, kind)

    if mlflow:
        with mlflow.start_run(run_name=kind):
            mlflow.set_tags({"lot": "P6", "task": "P6.2", "author": "wassima", "git_commit": git_commit(),
                             "features": "p4_features_dns + volumes P2"})
            params = IF_PARAMS if kind == "iforest" else HGB_PARAMS
            mlflow.log_params({k: v for k, v in params.items()} | {"target_fpr": TARGET_FPR,
                              "threshold_split": "val_benign"})
            for group, b in bundles.items():
                mlflow.log_param(f"{group}_n_features", len(b["columns"]))
                mlflow.log_metric(f"{group}/threshold", float(b["threshold"]))
                mlflow.log_text("\n".join(b["columns"]), f"features_{group}.txt")
                mlflow.log_artifact(str(MODELS_DIR / f"{kind}_{group}.joblib"), artifact_path="models")
            for r in rows:
                for k in ("recall", "fpr", "precision", "f1", "roc_auc", "pr_auc"):
                    v = r.get(k)
                    if v is not None and not np.isnan(v):
                        mlflow.log_metric(f"{r['split']}/{r['group']}/{k}", float(v))
    return scores, preds, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2", help="dossier output de P2 (sessions_pcap.parquet)")
    ap.add_argument("--supervised", action="store_true", help="ajoute le modèle supervisé de référence")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    print("[1/4] chargement sessions P2 + features P4")
    df = load_dataset(p2_dir(args.p2)).reset_index(drop=True)
    mlflow = None if args.no_mlflow else setup_mlflow("c2-isolation-forest")

    all_rows = []
    out = df[["session_id", "label", "split", "group", "family"]].copy()
    kinds = ["iforest"] + (["supervised_hgb"] if args.supervised else [])
    for i, kind in enumerate(kinds):
        print(f"[2/4] entraînement {kind}")
        scores, preds, rows = run(df, kind, mlflow)
        all_rows += rows
        out[f"{kind}_score"], out[f"{kind}_alert"] = scores, preds.astype("int8")
        if kind == "iforest":
            test = (df["split"] == "test").to_numpy()
            metrics_by(out[test].assign(pred=preds[test]), "family").to_csv(
                EVAL_DIR / "iforest_by_family.csv", index=False, float_format="%.6f")

    print("[3/4] sauvegarde")
    res = pd.DataFrame(all_rows)
    res.to_csv(EVAL_DIR / "iforest_metrics.csv", index=False, float_format="%.6f")
    (REPO / "p6_models" / "output").mkdir(parents=True, exist_ok=True)
    out.to_parquet(REPO / "p6_models" / "output" / "iforest_scores.parquet", index=False)

    print("[4/4] résumé :")
    view = res[res.split.isin(["val", "test", "unseen"])][["model", "split", "group", "n_c2", "n_benign",
                                                           "recall", "fpr", "roc_auc"]]
    print(view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
