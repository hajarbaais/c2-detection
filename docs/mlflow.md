# MLflow — suivi des expériences (P6)

MLflow enregistre, pour chaque entraînement : paramètres, métriques, seuils, liste des features,
modèle, et le commit Git. On peut ainsi comparer les modèles et refaire exactement un résultat.

## Installation et lancement

```bash
pip install -r requirements.txt          # contient mlflow
python -I p6_models/scripts/p6_01_rule_baselines.py --p2 <dossier output P2>
python -I p6_models/scripts/p6_02_isolation_forest.py --p2 <dossier output P2> --supervised
mlflow ui --backend-store-uri sqlite:///mlflow.db     # depuis la racine du dépôt
# puis ouvrir http://127.0.0.1:5000
```

Stockage local (gitignoré) : `mlflow.db` (runs, paramètres, métriques) + `mlruns/` (artefacts).
Pour un serveur commun plus tard : `set MLFLOW_TRACKING_URI=http://<serveur>:5000` (Windows) ou
`export MLFLOW_TRACKING_URI=...` — aucun changement de code.

## Conventions de l'équipe

Toujours passer par `setup_mlflow(...)` de `p6_models/scripts/p6_common.py` (même backend pour tous).

| Élément | Règle | Exemple |
|---|---|---|
| Expérience | `c2-<famille de modèle>` | `c2-baselines`, `c2-isolation-forest`, `c2-autoencoder`, `c2-lstm`, `c2-fusion` |
| Nom du run | modèle + variante | `rules_calibrated`, `iforest`, `lstm_jitter50` |
| Tags obligatoires | `lot`, `task`, `author`, `git_commit`, `features` | `P6`, `P6.2`, `wassima`, `a1b2c3d`, `p4_features_dns` |
| Paramètres | hyper-paramètres + `target_fpr` + split du seuil | `n_estimators=300`, `threshold_split=val_benign` |
| Métriques | `<split>/<groupe>/<métrique>` | `test/dns/recall`, `val/web/fpr`, `test/all/R_any/recall` |
| Artefacts | modèle + liste des features | `models/iforest_dns.joblib`, `features_dns.txt` |

Métriques minimum à logger : `recall`, `fpr` (objectif < 0,001), `precision`, `roc_auc`, `pr_auc`,
sur `val` et `test`, et le `recall` sur `unseen` (familles jamais vues).

Règles anti-fuite : seuil choisi sur la **validation** (ou le bénin du train), jamais sur le test ;
scaler / modèle ajustés sur le **train** uniquement.

## Exemple minimal (Oumaima / Imane / Hajar)

```python
from p6_common import setup_mlflow, git_commit
mlflow = setup_mlflow("c2-autoencoder")
with mlflow.start_run(run_name="ae_v1"):
    mlflow.set_tags({"lot": "P5", "task": "P5.6", "author": "oumaima", "git_commit": git_commit(),
                     "features": "p5_features_tls_seq"})
    mlflow.log_params({"hidden": 32, "epochs": 50, "lr": 1e-3})
    for epoch, loss in enumerate(losses):
        mlflow.log_metric("train/loss", loss, step=epoch)
    mlflow.log_metric("val/all/fpr", fpr)
    mlflow.log_artifact("p6_models/models/autoencoder.pt", artifact_path="models")
```
