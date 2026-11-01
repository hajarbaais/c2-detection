# P6 — Modèles & évaluation

## Entrées
Features de P4 + P5 (jointes sur `session_id`) + labels/splits de P2.

## Modèles
- **Isolation Forest** — baseline anomalie
- **Autoencoder** — anomalie, entraîné sur le bénin uniquement
- **LSTM / Transformer** — séquentiel (séquences de P5)

## Évaluation (`evaluation/`)
- **Recall** et **FPR par source et par famille** (jamais l'accuracy seule)
- Intervalles de confiance ; tester avec et sans `ctu13_broad_label`
- Explicabilité **SHAP**

## Cibles
Recall ≥ 98 %, FPR < 0,1 %, alerte < 5 s.

Les poids de modèles sont **gitignorés** (dossier `models/`).
