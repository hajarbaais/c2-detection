"""P6 - Enregistre dans MLflow les résultats du sprint 2 déjà calculés (p6_models/evaluation/*.csv).

Pourquoi : mlflow.db est gitignoré. Sur un nouveau clone ou un autre PC, la base MLflow est vide alors que
les CSV d'évaluation (versionnés) contiennent les vrais résultats. Ce script les remet dans MLflow pour
pouvoir les comparer dans l'interface, SANS relancer les entraînements (données P2 / P4 hors dépôt).

Runs créés (conventions de docs/mlflow.md, métriques <split>/<groupe>/<métrique>) :
  c2-baselines        rules_expert, rules_calibrated     <- baselines_metrics.csv   (p6_01, Wassima)
  c2-isolation-forest iforest, supervised_hgb            <- iforest_metrics.csv     (p6_02 --p5, Wassima)
  c2-autoencoder      ae_v2                              <- autoencoder_metrics.csv (p5_6b, Oumaima)

Chaque run porte le tag source=evaluation_csv : ce sont les métriques produites par les scripts
d'entraînement, pas un nouvel entraînement. Relancer le script ne crée pas de doublon.

Usage (depuis la racine du dépôt) :
    python -I p6_models/scripts/p6_log_results_mlflow.py
    python -m mlflow server --workers 1        # puis http://127.0.0.1:5000
"""
import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_common import EVAL_DIR, TARGET_FPR, git_commit, setup_mlflow  # noqa: E402

METRICS = ("recall", "fpr", "precision", "f1", "roc_auc", "pr_auc")
IF_PARAMS = dict(n_estimators=300, max_samples=256, contamination="auto", random_state=42)   # p6_02
HGB_PARAMS = dict(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=42)  # p6_02
EXPERT_THR = {"thr_R1_entropie": 3.5, "thr_R2_longueur": 40, "thr_R3_periodicite": 0.5,
              "thr_R4_regularite": -0.2}                                                      # p6_01


def log_rows(mlflow, rows: pd.DataFrame, with_rule: bool = False):
    for _, r in rows.iterrows():
        prefix = f"{r['split']}/{r['group']}/" + (f"{r['rule']}/" if with_rule else "")
        for k in METRICS:
            v = r.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                mlflow.log_metric(prefix + k, float(v))


def already_logged(mlflow, experiment: str, run_name: str) -> bool:
    runs = mlflow.search_runs(experiment_names=[experiment],
                              filter_string=f"tags.source = 'evaluation_csv' and tags.mlflow.runName = '{run_name}'")
    return len(runs) > 0


def new_run(mlflow, experiment, run_name, tags, params, csv, artifacts):
    mlflow.set_experiment(experiment)
    if already_logged(mlflow, experiment, run_name):
        print(f"  = {experiment}/{run_name} déjà présent, ignoré")
        return None
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tags(tags | {"git_commit": git_commit(), "source": "evaluation_csv",
                            "source_file": f"p6_models/evaluation/{csv}"})
    mlflow.log_params(params | {"target_fpr": TARGET_FPR, "threshold_split": "val_benign"})
    if artifacts:
        mlflow.log_artifact(str(EVAL_DIR / csv))
    print(f"  + {experiment}/{run_name}")
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-artifacts", action="store_true", help="ne copie pas les CSV comme artefacts")
    a = ap.parse_args()
    mlflow = setup_mlflow("c2-baselines")
    if mlflow is None:
        sys.exit("[!] mlflow non installé : pip install mlflow")
    for exp in ("c2-isolation-forest", "c2-autoencoder"):
        setup_mlflow(exp)
    art = not a.no_artifacts

    # 1. Baselines à règles (même filtre que p6_01 : groupe all, ou règle R_any)
    base = pd.read_csv(EVAL_DIR / "baselines_metrics.csv")
    for mode in ("expert", "calibrated"):
        params = {"mode": mode} | (EXPERT_THR if mode == "expert" else {})
        if new_run(mlflow, "c2-baselines", f"rules_{mode}",
                   {"lot": "P6", "task": "P6.1", "author": "wassima", "features": "p4_features_dns"},
                   params, "baselines_metrics.csv", art):
            rows = base[(base["mode"] == mode) & ((base["group"] == "all") | (base["rule"] == "R_any"))]
            log_rows(mlflow, rows, with_rule=True)
            mlflow.end_run()

    # 2. Isolation Forest (+ supervisé de référence), web = features TLS de P5
    ifo = pd.read_csv(EVAL_DIR / "iforest_metrics.csv")
    for kind, params in (("iforest", IF_PARAMS), ("supervised_hgb", HGB_PARAMS)):
        if new_run(mlflow, "c2-isolation-forest", kind,
                   {"lot": "P6", "task": "P6.2", "author": "wassima",
                    "features": "p4_features_dns + p5_tls (rareté sur bénin)"},
                   params, "iforest_metrics.csv", art):
            log_rows(mlflow, ifo[ifo["model"] == kind])
            mlflow.end_run()

    # 3. Autoencodeur d'Oumaima (version retenue ae_v2, split de P2)
    ae = pd.read_csv(EVAL_DIR / "autoencoder_metrics.csv")
    if new_run(mlflow, "c2-autoencoder", "ae_v2",
               {"lot": "P5", "task": "P5.6", "author": "oumaima", "experiment": "split_P2",
                "features": "p4_features_dns + p5_tls (rareté sur bénin)"},
               {"epochs_max": 300, "patience": 20}, "autoencoder_metrics.csv", art):
        log_rows(mlflow, ae)
        mlflow.end_run()
    print("OK. Interface : python -m mlflow server --workers 1  ->  http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
