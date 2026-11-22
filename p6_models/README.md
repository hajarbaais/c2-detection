# P6 — Modèles & évaluation

## Entrées
Features de P4 + P5 (jointes sur `session_id`, **jointure gauche** depuis `sessions_pcap.parquet`) + labels/splits de P2.

## Modèles
- **Baselines à règles** (P6.1) — `scripts/p6_01_rule_baselines.py` ✅ sprint 2
- **Isolation Forest** (P6.2) — `scripts/p6_02_isolation_forest.py` ✅ sprint 2 (+ `--supervised` : HistGradientBoosting de référence ; `--p5` : features TLS pour le web)
- **Autoencoder** (P5.6, Oumaima) — `scripts/p5_6b_autoencoder_train.py`, anomalie, entraîné sur le bénin uniquement ; branché dans le pipeline P7
- **LSTM** (P6.3, Wassima) — `scripts/p6_03_lstm.py` — sprint 3, + **robustesse au jitter** (P6.4) `scripts/p6_04_jitter.py` → `README_lstm.md`
- **Transformer** (P6.5, Imane) — `scripts/p6_05_transformer.py` — sprint 3 → `README_transformer.md`
- **Comparaison LSTM / Transformer** — `scripts/p6_compare_seq.py` → `comparison_lstm_transformer.md`

## Lancer (sprint 2)

```bash
# 0. features P4 (Hajar) — une fois
for s in p4_01_entropy p4_02_shape p4_03_iat p4_04_periodicity p4_05_windows; do
  python -I p4_features_dns_time/scripts/$s.py <dossier output P2>; done
# 1. baselines à règles (modes expert + calibré)
python -I p6_models/scripts/p6_01_rule_baselines.py --p2 <dossier output P2>
# 2. Isolation Forest (+ modèle supervisé de référence) avec les features TLS de P5
#    (écrit aussi models/tls_reference.joblib, utilisé par le pipeline)
python -I p6_models/scripts/p6_02_isolation_forest.py --p2 <dossier output P2> --p5 <dossier p5_output> --supervised
# 3. résultats
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Le dossier P2 peut aussi être donné une fois pour toutes : `set C2_P2_DIR=...` (Windows).

| Fichier | Rôle |
|---|---|
| `scripts/p6_common.py` | chargement P2 + P4 (+ P5), choix des features autorisées, MLflow |
| `scripts/p6_tls.py` | features P5 : jointure + rareté JA3/JA4 et réputation recalculées sur le **bénin du train web** (référence sauvegardée) |
| `evaluation/metrics.py` | recall, FPR, IC de Wilson, seuil à FPR cible, métriques par famille |
| `evaluation/*_metrics.csv`, `*_by_family.csv` | résultats (versionnés, petits) |
| `models/` | `iforest_<dns|web>.joblib`, `rules_thresholds.json`, `tls_reference.joblib`, `autoencoder.pt` (gitignorés, à partager par Drive) |

## Méthode
- Un modèle par groupe : `dns` (contenu DNS + rythme) et `web` (rythme + 14 features TLS de P5, mêmes colonnes que l'autoencodeur).
- Features P5 : `ja3_rarity`, `ja4_rarity` et `log_domain_reputation` de `features_tls.parquet` sont calculées sur tout le
  train, C2 compris ; elles sont **recalculées sur le bénin du train web** (`p6_tls.py`, même formule que P5.6b).
- Isolation Forest entraîné **sur le bénin du train uniquement** ; seuil fixé sur le **bénin de validation** à 0,1 % de FPR.
- Règles calibrées : seuil = quantile du bénin du train (budget de FPR partagé entre les 4 règles).
- Le test n'est jamais utilisé pour choisir quoi que ce soit.

## Résultats sprint 2 (seuil à 0,1 % de FPR sur le bénin de validation)

Features P4 régénérées avec le correctif de Hajar (`42a089c` : `family` dans p4_04, regroupement par
`(file, session_key)` dans p4_05). Les chiffres sont donc différents de la première version de ce tableau.
Le pipeline reproduit exactement les métriques de l'autoencodeur d'Oumaima (vérifié sur val et test).

| Modèle | Recall val | Recall test | FPR test | Recall unseen | ROC-AUC test (dns / web) |
|---|---|---|---|---|---|
| Règles expertes (R_any) | 1,8 % | 1,4 % | 1,92 % | 23,1 % | – |
| Règles calibrées (R_any) | 1,1 % | 0,4 % | 0,036 % | 0,9 % | – |
| Isolation Forest, web = volumes P2 | 1,7 % | 0,8 % | 0,81 % | 0,5 % | 0,83 / 0,91 |
| **Isolation Forest, web = TLS P5** | 2,8 % | 0,4 % | **0,14 %** | 0,5 % | 0,83 / **0,95** |
| Autoencodeur `ae_v2` (Oumaima) | – | dns 3,7 % / web 55,8 % | dns 0,15 % / web 4,13 % | dns 27,9 % / web 0,7 % | **0,94** / 0,91 |
| HistGradientBoosting (supervisé, référence, TLS P5) | 0,8 % | 39,9 % | 0,003 % | 19,9 % | 1,00 / 0,92 |

Détail web de l'Isolation Forest (test) : avec les features TLS, le classement s'améliore (ROC-AUC 0,905 → 0,949) et la
FPR test passe de 1,08 % à 0,15 % (835 → 113 fausses alertes), mais le rappel au seuil reste quasi nul (0,04 %).

**Expérience avec le redécoupage du bénin web d'Oumaima** (`resplit_benign_web`, borne optimiste, non versionnée) :
Isolation Forest web + TLS P5 → val 13 906 bénins, **FPR test 0,086 %** (objectif atteint), ROC-AUC 0,975, mais
rappel test 2,6 %. Sans P5 : FPR 0,092 %, rappel 0,1 %.

Lecture :
- Les règles détectent **100 %** de l'exfiltration DNS-EXF (7 sessions de test, entropie / longueur) mais 0 % des balises
  (trickbot, dridex) et du C2 « large » CTU-13 → il faut le modèle séquentiel (sprint 3).
- Les features TLS de P5 aident l'Isolation Forest à **classer** (AUC 0,95) et font chuter les fausses alertes, mais au seuil
  de 0,1 % il ne garde presque rien : c'est un détecteur de référence, pas le détecteur final.
- Le seuil web reste fragile tant que la validation web n'a que 2 268 bénins (0,1 % = 2 sessions) : voir
  `docs/demande_P2_rebalancement.md`. Avec un découpage équilibré, la FPR test de l'IF tombe sous 0,1 %.
- HistGradientBoosting web : sa probabilité sature à 1,0 sur quelques bénins de validation → seuil = 1,0 et rappel nul.
  Le supervisé apprend surtout des raccourcis de dataset (`tls_version`, `has_ja3`, sessions `http` = 91 % C2) : à ne
  pas présenter comme un résultat, seulement comme une borne de comparaison.

## Cibles
Recall ≥ 98 %, FPR < 0,1 %, alerte < 5 s. Les poids de modèles sont **gitignorés**.
