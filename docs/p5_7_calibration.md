# P5.7 — Calibration du seuil de l'autoencodeur à 0,1 % de FPR (Oumaima)

- **Tâche** : P5.7, Sprint 3 — « Seuil AE exact + recall obtenu »
- **Script** : `p6_models/scripts/p5_7_calibration.py` — **Date** : 22/11/2026 — **Branche** : `feature/oumaima-sprint2`
- **Entrées** : `autoencoder.pt` (run `ae_v2`, découpage de P2) et `autoencoder_resplit.pt` (expérience de redécoupage du bénin web)

| Critère de la tâche | État |
|---|---|
| Seuil calculé | Fait : `p6_models/evaluation/ae_thresholds.json`, `ae_calibration.csv` (par groupe, mode et FPR cible) |
| Recall mesuré | Fait, avec intervalles de confiance à 95 %, par groupe et par famille |
| Score par session | Fait : `p6_models/output/ae_scores.parquet` (non versionné, à partager par Drive) |

## 1. Méthode

Pour chaque groupe (`dns`, `web`) :
1. Score (erreur de reconstruction) de toutes les sessions avec le modèle entraîné.
2. **Seuil sur le bénin de validation** : le plus petit seuil laissant au plus `floor(0,1 % × n)` faux positifs. Alerte si score > seuil.
3. **FPR et recall mesurés sur le test**, qui n'intervient jamais dans un choix, avec intervalles de Wilson à 95 %.
4. **Fiabilité du seuil** : avec `n` bénins en validation, 0,1 % autorise `n/1000` fausses alertes. En dessous de 10 (`n < 10 000`), le seuil est jugé *peu fiable*.
5. **Seuil « strict »** (règle des trois) quand `n ≥ 3 000` : seuil sans aucun faux positif en validation (0 faux positif sur `n` sessions donne un FPR < 3/n à 95 %).
6. Compromis à 0,5 % et 1 % de FPR, rappel par famille, DNS **sans les requêtes DNS banales de CTU-13** (`ctu13_broad_label`, option `--p3`).
7. Contrôle : le seuil recalculé est égal à celui stocké dans le `.pt` (écart nul pour les deux groupes et les deux modèles), donc les features n'ont pas changé depuis l'entraînement. `ae_scores.parquet` : 184 979 lignes, aucun score manquant ; alertes du test cohérentes avec les tableaux (DNS : 111 = 44 + 67 ; web : 4 512 = 3 202 + 1 310).

## 2. Résultats au seuil de 0,1 % (objectif du professeur : FPR < 0,1 % et recall ≥ 98 %)

| Groupe | Découpage | Bénins val | Fausses alertes autorisées | Seuil | FPR test [IC 95 %] | Recall test [IC 95 %] |
|---|---|---|---|---|---|---|
| dns | P2 | 9 689 | 9 (**peu fiable**) | 0,4993 | 0,145 % [0,108 – 0,195] (44 / 30 312) | 3,70 % [2,92 – 4,67] (67 / 1 811) |
| web | P2 | 2 268 | 2 (**peu fiable**) | 0,7490 | **4,13 %** [3,99 – 4,27] (3 202 / 77 591) | 55,8 % [53,8 – 57,8] (1 310 / 2 346) |
| web | expérience (re-split) | 13 906 | 13 (fiable) | 0,3638 | **0,56 %** [0,46 – 0,69] (92 / 16 303) | 59,6 % [57,6 – 61,5] (1 397 / 2 346) |

**Les objectifs ne sont atteints dans aucun cas** : ni FPR < 0,1 %, ni recall ≥ 98 %.

## 3. Compromis recall / FPR (seuils calibrés sur le bénin de validation, mesures sur le test)

| Groupe | Mode | FPR cible (val) | FPR test | Recall test |
|---|---|---|---|---|
| dns | quantile | 0,1 % | 0,15 % | 3,7 % |
| dns | quantile | 0,5 % | 0,64 % | 13,5 % |
| dns | quantile | 1 % | 1,06 % | 26,3 % |
| dns | strict (règle des trois) | 0 faux positif | 0,003 % (1 / 30 312) | 0,06 % (1 / 1 811) |
| web (P2) | quantile | 0,1 % / 0,5 % / 1 % | 4,13 % / 5,26 % / 6,09 % | 55,8 % / 56,0 % / 59,5 % |
| web (re-split) | quantile | 0,1 % | 0,56 % | 59,6 % |
| web (re-split) | quantile | 0,5 % | 3,93 % | 71,0 % |
| web (re-split) | quantile | 1 % | 5,37 % | 72,0 % |
| web (re-split) | strict (règle des trois) | 0 faux positif | **0 / 16 303** (borne haute 0,02 %) | **10,2 %** [9,1 – 11,5] (240 / 2 346) |

Le seul réglage qui respecte FPR < 0,1 % sur le test est le seuil strict, au prix d'un recall de 10,2 % (web, découpage
expérimental) ou de 0,06 % (DNS). Pour le web avec le découpage de P2, le seuil strict n'est pas calculable : 2 268 bénins
en validation, moins de 3 000.

## 4. DNS sans les requêtes DNS banales de CTU-13

| | C2 | Détectés | Recall [IC 95 %] |
|---|---|---|---|
| Seuils à 0,1 %, 0,5 % et 1 % | 7 | 7 | **100 %** [64,6 – 100] |
| Seuil strict | 7 | 0 | 0 % |

Sur la vraie exfiltration DNS, le modèle détecte 7 sessions sur 7, mais l'effectif est minuscule. Le seuil strict (10,24) est fixé par une seule
session bénigne de validation au score extrême, supérieur à celui de toutes les sessions d'exfiltration : ce seuil dépend d'une valeur aberrante.

## 5. Par famille (web, test, seuil à 0,1 %)

| Famille | Sessions | Recall (P2) | Recall (re-split) |
|---|---|---|---|
| trickbot | 1 469 | 72,5 % | 77,9 % |
| dridex | 600 | 40,0 % | 41,2 % |
| cobaltstrike | 5 | 60,0 % (3 / 5) | 0 % (0 / 5) |
| neris | 184 | 0,5 % | 2,2 % |
| virut | 37 | 2,7 % | 2,7 % |
| rbot, dns2tcp, iodine | 12, 19, 20 | 0 % | 0 % |

| Famille bénigne | Sessions test (P2 / re-split) | FPR (P2) | FPR (re-split) |
|---|---|---|---|
| benign_https | 75 735 / 15 909 | 3,41 % | 0,57 % |
| benign_doh | 999 / 214 | 61,6 % | 0 % |
| benign_normal_user | 857 / 180 | 0,35 % | 1,1 % |

## 6. Analyse

1. **Desserrer le seuil ne corrige pas le web.** Avec le découpage de P2, passer de 0,1 % à 1 % de FPR cible fait monter le FPR de
   test de 4,13 % à 6,09 % pour un recall qui passe de 55,8 % à 59,5 % : le défaut vient de la différence de distribution entre la
   validation et le test, pas du choix du seuil.
2. **Le re-split réduit nettement l'écart** (FPR de test 0,56 % au lieu de 4,13 %), mais pas jusqu'à 0,1 %. À 0,5 % de budget, le FPR
   de test saute à 3,93 % : en abaissant le seuil, 548 fausses alertes de plus apparaissent entre les scores 0,21 et 0,36. Hypothèse (non
   vérifiée) : un groupe de sessions bénignes du test, absent de la validation, par exemple la fin des captures.
3. **Le rappel web est limité à trickbot et dridex**, avec ou sans re-split : les autres familles ressemblent au bénin dans l'espace des
   features. Ce n'est pas un problème de seuil.
4. **DNS** : le recall global (3,7 %) est dominé par les 1 804 sessions CTU-13 « banales », labellisées C2 par convention (voir `docs/p5_6b_autoencoder_train.md`).
5. **Le compromis réel est donc** : FPR < 0,1 % avec un recall de 10 % (web, découpage expérimental) ou recall de ~60–72 % avec un
   FPR de 0,6 % à 5 %.

## 7. Limites

- Seuil DNS **peu fiable** (9 fausses alertes autorisées) et seuil web de P2 **peu fiable** (2). Le seuil web du re-split est fiable, mais le test est **optimiste** : mêmes machines, sessions voisines dans le temps (voir `docs/p5_6b_autoencoder_train.md`, section 6 bis).
- Un seul modèle par groupe, une seule graine ; les intervalles de Wilson ne mesurent pas la variabilité due à l'entraînement.
- Recall et FPR mesurés sur un seul test ; `unseen` non utilisé ici (aucun bénin).
- Le recall par famille repose parfois sur de très petits effectifs (cobaltstrike : 5 sessions).
- La calibration reste valable seulement pour le couple (modèle, découpage, features) utilisé ; elle doit être refaite après tout réentraînement ou nouveau découpage de P2.

## 8. Reproduire

```bash
python -I p6_models/scripts/p5_7_calibration.py --p2 <dossier sessions_pcap.parquet> --p5 <dossier p5_output> --p3 <sessions_unified.parquet>
python -I p6_models/scripts/p5_7_calibration.py --p2 ... --p5 ... --p3 ... --resplit-benign-web     # modèle autoencoder_resplit.pt
```
Sorties : `p6_models/evaluation/ae_calibration{,_resplit}.csv`, `ae_calibration_by_family{,_resplit}.csv`, `ae_thresholds{,_resplit}.json`,
`p6_models/output/ae_scores{,_resplit}.parquet` (non versionné).

## 9. Suite : fusion des scores (P6.6)

`ae_scores.parquet` donne le score et le seuil de chaque session. La fusion avec le modèle séquentiel (LSTM, P6.3) attend les scores
de Wassima (`lstm_scores.parquet`, hors Git). La règle devra gérer les sessions DNS, qui n'ont que ~2 pas en moyenne.
