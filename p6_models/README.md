# P6 — Modèles & évaluation

## Entrées
Features de P4 + P5 (jointes sur `session_id`, **jointure gauche** depuis `sessions_pcap.parquet`) + labels/splits de P2.

## Modèles
- **Baselines à règles** (P6.1) — `scripts/p6_01_rule_baselines.py` ✅ sprint 2
- **Isolation Forest** (P6.2) — `scripts/p6_02_isolation_forest.py` ✅ sprint 2 (+ `--supervised` : HistGradientBoosting de référence)
- **Autoencoder** — anomalie, entraîné sur le bénin uniquement (P5.6)
- **LSTM / Transformer** — séquentiel (séquences de P5) — sprint 3

## Lancer (sprint 2)

```bash
# 0. features P4 (Hajar) — une fois
for s in p4_01_entropy p4_02_shape p4_03_iat p4_04_periodicity p4_05_windows; do
  python -I p4_features_dns_time/scripts/$s.py <dossier output P2>; done
# 1. baselines à règles (modes expert + calibré)
python -I p6_models/scripts/p6_01_rule_baselines.py --p2 <dossier output P2>
# 2. Isolation Forest (+ modèle supervisé de référence)
python -I p6_models/scripts/p6_02_isolation_forest.py --p2 <dossier output P2> --supervised
# 3. résultats
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Le dossier P2 peut aussi être donné une fois pour toutes : `set C2_P2_DIR=...` (Windows).

| Fichier | Rôle |
|---|---|
| `scripts/p6_common.py` | chargement P2 + P4, choix des features autorisées, MLflow |
| `evaluation/metrics.py` | recall, FPR, IC de Wilson, seuil à FPR cible, métriques par famille |
| `evaluation/*_metrics.csv`, `*_by_family.csv` | résultats (versionnés, petits) |
| `models/` | `iforest_<dns|web>.joblib`, `rules_thresholds.json` (gitignorés) |

## Méthode
- Un modèle par groupe : `dns` (contenu DNS + rythme) et `web` (rythme + volumes P2 ; TLS de P5 à ajouter).
- Isolation Forest entraîné **sur le bénin du train uniquement** ; seuil fixé sur le **bénin de validation** à 0,1 % de FPR.
- Règles calibrées : seuil = quantile du bénin du train (budget de FPR partagé entre les 4 règles).
- Le test n'est jamais utilisé pour choisir quoi que ce soit.

## Résultats sprint 2 (seuil à 0,1 % de FPR)

| Modèle | Recall val | Recall test | FPR test | Recall unseen |
|---|---|---|---|---|
| Règles expertes (R_any) | 1,8 % | 1,4 % | 1,92 % | 23,1 % |
| Règles calibrées (R_any) | 1,1 % | 0,4 % | 0,036 % | 0,9 % |
| Isolation Forest | 1,5 % | 3,1 % | 0,77 % | 0 % |
| HistGradientBoosting (supervisé, référence) | 36,4 % | 77,8 % | 1,05 % | 66,1 % |

ROC-AUC test de l'Isolation Forest : 0,80 (dns), 0,92 (web) — il classe plutôt bien, mais au seuil très
strict de 0,1 % de FPR il ne garde presque rien.

Lecture :
- Les règles détectent **100 %** de l'exfiltration DNS-EXF (entropie / longueur) mais **0 %** des balises
  (trickbot, dridex) et du C2 « large » CTU-13 → c'est bien la preuve qu'il faut le modèle séquentiel.
- La FPR du test est plus haute que celle de la validation : le bénin web de la validation (2 268 sessions)
  est trop petit pour calibrer 0,1 % (= 2 sessions) et le test contient 76 000 sessions DoHBrw bénignes
  d'une autre distribution. À corriger avec P2 : plus de bénin web en validation.
- Le web n'a pour l'instant que rythme + volumes bruts : les features TLS de P5 (JA3/JA4, réputation) devraient beaucoup aider.

## Cibles
Recall ≥ 98 %, FPR < 0,1 %, alerte < 5 s. Les poids de modèles sont **gitignorés**.
