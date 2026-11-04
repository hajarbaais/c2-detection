"""P6.1 - Baselines à règles : le point de comparaison « pare-feu classique ».

Idée : avant tout ML, on mesure ce que donnent des seuils simples sur les features P4.
Les modèles (Isolation Forest, LSTM, fusion) devront faire MIEUX que ces règles.

Règles (score élevé = suspect ; alerte si score > seuil) :
  R1_entropie     dns_entropy_max          > 3.5 bits      sous-domaine encodé (base32/64)
  R2_longueur     dns_qname_len_max        > 40 caractères nom anormalement long (tunnel)
  R3_periodicite  dns_periodicity_score    > 0.5           rythme d'horloge (beaconing)
  R4_regularite   -dns_iat_cv              > -0.2          IAT très réguliers (CV < 0.2), si >= 10 événements
  R_any           OU logique des règles ci-dessus

Deux modes, enregistrés comme deux runs MLflow :
  expert     seuils fixés à la main ci-dessus (règle « à la Snort »)
  calibrated seuil de chaque règle = quantile du score sur le BÉNIN DU TRAIN pour 0,1 % de FPR
             (aucune donnée de val/test utilisée -> pas de fuite)

Sorties :
  p6_models/evaluation/baselines_metrics.csv      recall / FPR / IC par règle, mode, split
  p6_models/evaluation/baselines_by_family.csv    recall / FPR par famille (test, R_any)
  p6_models/output/baseline_scores.parquet        scores par session (gitignoré)
  p6_models/models/rules_thresholds.json          seuils calibrés (utilisés par le pipeline P7)

Usage : python -I p6_01_rule_baselines.py [--p2 DOSSIER_OUTPUT_P2] [--no-mlflow]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from p6_common import EVAL_DIR, MODELS_DIR, REPO, TARGET_FPR, git_commit, load_dataset, p2_dir, setup_mlflow  # noqa: E402
from metrics import binary_metrics, metrics_by, threshold_at_fpr  # noqa: E402

MIN_EVENTS_REGULARITY = 10

# name, colonne, sens (+1 : grand = suspect ; -1 : petit = suspect), seuil expert (sur le score signé)
RULES = [
    ("R1_entropie", "dns_entropy_max", +1, 3.5),
    ("R2_longueur", "dns_qname_len_max", +1, 40.0),
    ("R3_periodicite", "dns_periodicity_score", +1, 0.5),
    ("R4_regularite", "dns_iat_cv", -1, -0.2),
]


def rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Score signé de chaque règle (NaN = règle non applicable -> jamais d'alerte)."""
    out = pd.DataFrame(index=df.index)
    for name, col, sign, _ in RULES:
        s = sign * df[col].astype("float64")
        if name == "R4_regularite":
            s = s.where(df["n_events"] >= MIN_EVENTS_REGULARITY)
        out[name] = s
    return out


def thresholds(scores: pd.DataFrame, df: pd.DataFrame, mode: str) -> dict:
    if mode == "expert":
        return {name: thr for name, _, _, thr in RULES}
    benign_train = (df["split"] == "train") & (df["label"] == 0)
    # avec 4 règles en OU, chaque règle reçoit 1/4 du budget de FPR (correction de Bonferroni)
    return {name: threshold_at_fpr(scores.loc[benign_train, name].to_numpy(), TARGET_FPR / len(RULES))
            for name, *_ in RULES}


def predict(scores: pd.DataFrame, thr: dict) -> pd.DataFrame:
    pred = pd.DataFrame({name: (scores[name] > thr[name]).fillna(False) for name in thr})
    pred["R_any"] = pred.any(axis=1)
    return pred


def evaluate(df, scores, pred, mode) -> list[dict]:
    rows = []
    for split in ["train", "val", "test", "unseen"]:
        m_split = df["split"] == split
        for rule in list(pred.columns):
            for grp in ["all", "dns", "web"]:
                m = m_split & ((df["group"] == grp) if grp != "all" else True)
                if not m.any():
                    continue
                sc = scores.loc[m, rule] if rule in scores.columns else None   # R_any : pas de score continu
                met = binary_metrics(df.loc[m, "label"], pred.loc[m, rule], sc)
                rows.append({"mode": mode, "rule": rule, "split": split, "group": grp, **met})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2", help="dossier output de P2 (sessions_pcap.parquet)")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    print("[1/4] chargement sessions P2 + features P4")
    df = load_dataset(p2_dir(args.p2))
    scores = rule_scores(df)
    print(f"       {len(df):,} sessions ({(df.label == 1).sum():,} C2)")

    mlflow = None if args.no_mlflow else setup_mlflow("c2-baselines")
    all_rows, out_scores = [], df[["session_id", "label", "split", "group"]].copy()
    for mode in ["expert", "calibrated"]:
        print(f"[2/4] mode {mode}")
        thr = thresholds(scores, df, mode)
        pred = predict(scores, thr)
        rows = evaluate(df, scores, pred, mode)
        all_rows += rows
        out_scores[f"{mode}_R_any"] = pred["R_any"].astype("int8")
        for r in rows:
            if r["rule"] == "R_any" and r["group"] == "all" and r["split"] in ("val", "test"):
                print(f"       {r['split']:5} R_any : recall={r['recall']:.3f}  FPR={r['fpr']:.5f} "
                      f"(IC95 {r['fpr_ci_low']:.5f}-{r['fpr_ci_high']:.5f})")
        if mlflow:
            with mlflow.start_run(run_name=f"rules_{mode}"):
                mlflow.set_tags({"lot": "P6", "task": "P6.1", "author": "wassima", "git_commit": git_commit(),
                                 "features": "p4_features_dns"})
                mlflow.log_params({f"thr_{k}": round(v, 6) for k, v in thr.items()} |
                                  {"mode": mode, "target_fpr": TARGET_FPR})
                for r in rows:
                    if r["group"] == "all" or r["rule"] == "R_any":
                        for k in ("recall", "fpr", "precision", "f1", "roc_auc", "pr_auc"):
                            v = r.get(k)
                            if v is not None and not np.isnan(v):
                                mlflow.log_metric(f"{r['split']}/{r['group']}/{r['rule']}/{k}", float(v))
        if mode == "calibrated":
            # seuils calibrés réutilisés par le pipeline P7 (p7_app_demo/pipeline.py)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            (MODELS_DIR / "rules_thresholds.json").write_text(json.dumps(thr, indent=2))
            test = df["split"] == "test"
            fam = metrics_by(df[test].assign(pred=pred.loc[test, "R_any"]), "family")
            fam.to_csv(EVAL_DIR / "baselines_by_family.csv", index=False)

    print("[3/4] sauvegarde des métriques")
    res = pd.DataFrame(all_rows)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(EVAL_DIR / "baselines_metrics.csv", index=False, float_format="%.6f")
    out_dir = REPO / "p6_models" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_scores.join(scores).to_parquet(out_dir / "baseline_scores.parquet", index=False)

    print("[4/4] résumé (test, toutes sessions) :")
    view = res[(res.split == "test") & (res.group == "all")][["mode", "rule", "recall", "fpr", "precision"]]
    print(view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
