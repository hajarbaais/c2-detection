# P6.6 — Fusion des scores : autoencodeur + modèle séquentiel (Oumaima)

- **Tâche** : P6.6, Sprint 3 — « Score final AE + séquentiel + verdict, `scores.parquet` »
- **Scripts** : `p6_models/scripts/p6_07_fusion.py` (fusion) et `p6_07a_export_lstm_scores.py` (export des scores du LSTM) — **Date** : 25/11/2026
- **Scénario retenu** : `ae+transformer`, règle « et », sessions web sans séquence **non évaluées** (`--noseq abstain`)

| Critère de la tâche | État |
|---|---|
| Règle de fusion | Faite : `p6_models/evaluation/fusion_rule.json` ; 4 règles et 15 combinaisons comparées dans `fusion_metrics.csv` |
| `scores.parquet` | Fait : `p6_models/output/scores.parquet` (184 979 sessions, non versionné, à partager par Drive) |
| Verdict | `ALERTE` / `NORMAL` / `NON_EVALUE` par session ; `apply_fusion()` pour le pipeline |

## 1. Entrées

| Détecteur | Fichier | Origine | Sessions avec score valide |
|---|---|---|---|
| Autoencodeur (`ae`) | `ae_scores.parquet` | P5.7, modèle `ae_v2` (`autoencoder.pt`) | 184 979 (dns et web) |
| Transformer | `transformer_scores_packets.parquet` | Imane, `tf_web_anomaly_packets.pt` | 89 236 (web avec paquets) |
| LSTM | `lstm_scores.parquet` | `p6_07a_export_lstm_scores.py` sur `lstm_web_anomaly_packets.pt` (Wassima) | 89 236 (web avec paquets) |

Les scores du LSTM ne sont pas fournis par Wassima : ils sont recalculés avec son code (`load_lstm`, `score_lstm`) et ses `features_seq.npz` et
`seq_scaler.pkl`. **Contrôle réussi** : le test web retrouve exactement ses chiffres (1 874 C2 / 75 103 bénins, rappel 42,16 %, FPR 0,29 %, 218 fausses alertes).
Les modèles séquentiels n'existent que pour le web avec de vrais paquets (les sessions « fallback » et vides n'ont pas de score) ; aucun modèle séquentiel utile n'existe pour le DNS.
Tous les modèles reposent sur le découpage officiel de P2.

## 2. Méthode

1. **Rangs** : chaque score est converti en rang par rapport au bénin de **validation** de sa population (ECDF), ce qui rend les échelles comparables.
   Au-delà du maximum de validation, le rang est prolongé d'un terme strictement croissant (sinon plusieurs sessions deviendraient ex æquo).
2. **Règles** : `max` (l'un ou l'autre est extrême), `mean`, `and` (les deux doivent l'être, minimum des rangs), `fisher` (somme de −log(1−u)).
3. **Strates**, chacune calibrée séparément :
   - `dns` : autoencodeur seul ;
   - `web_seq` : web avec séquence valide, fusion autoencodeur + séquentiel ;
   - `web_noseq` : web sans séquence valide : autoencodeur seul, ou **abstention** (`--noseq abstain`, aucune alerte, verdict `NON_EVALUE`).
4. **Seuil** : bénin de validation, FPR cible 0,1 %. Le test n'intervient dans aucun choix.
5. **Candidats comparés** : chaque séquentiel avec l'autoencodeur, puis tous ensemble, avec les 4 règles ; **et les détecteurs seuls** sur la même population, calibrés de la même façon.
6. **Sélection automatique** : meilleur recall de validation de la strate `web_seq`. Elle peut être forcée (`--select 'ae+transformer|and'`).

## 3. Résultats

### 3.1 Candidats sur la strate `web_seq` (test : 1 874 C2, 75 103 bénins ; validation : 2 170 bénins, 2 fausses alertes autorisées)

| Configuration | Règle | Recall validation | FPR test | Recall test |
|---|---|---|---|---|
| ae+transformer | max / mean / and / fisher | 84,2 / 56,9 / 55,1 / 58,2 % | 3,38 / 0,24 / **0,14** / 1,11 % | 74,4 / 46,0 / 41,7 / 58,9 % |
| ae+lstm | max / mean / and / fisher | 83,0 / 55,2 / 55,3 / 56,0 % | 3,35 / 0,20 / 0,21 / 0,93 % | 70,1 / 42,8 / 41,8 / 54,6 % |
| ae+transformer+lstm | max / mean / and / fisher | 84,8 / 55,1 / 55,1 / 58,1 % | 3,43 / 0,19 / 0,14 / 0,38 % | 74,4 / 41,7 / 41,7 / 48,1 % |
| **Détecteurs seuls** : autoencodeur | – | 79,5 % | 4,26 % | 69,9 % |
| Transformer | – | 60,1 % | 0,32 % | 46,5 % |
| LSTM | – | 58,9 % | 0,29 % | 42,2 % |

### 3.2 Scénario retenu : `ae+transformer`, règle « et », abstention (seuil 0,1 % sur la validation)

| Strate | Bénins val | Fausses alertes autorisées | FPR test [IC 95 %] | Recall test [IC 95 %] |
|---|---|---|---|---|
| dns (autoencodeur seul) | 9 689 | 9 (peu fiable) | 0,145 % [0,11 – 0,19] (44 / 30 312) | 3,70 % [2,9 – 4,7] (67 / 1 811) |
| web_seq (fusion) | 2 170 | 2 (peu fiable) | **0,14 %** [0,12 – 0,17] (105 / 75 103) | **41,7 %** [39,5 – 43,9] (781 / 1 874) |
| web_noseq (non évalué) | 98 | 0 | 0 (0 / 2 488) | 0 % (0 / 472) : sessions non évaluées, comptées comme manquées |
| **Web (agrégé)** | 2 268 | 2 | 0,14 % [0,11 – 0,16] | 33,3 % [31,4 – 35,2] (781 / 2 346) |
| **Tout (agrégé)** | 11 957 | 11 | **0,14 %** [0,12 – 0,16] (149 / 107 903) | **20,4 %** [19,2 – 21,7] (848 / 4 157) |

**Les objectifs du professeur ne sont atteints** ni pour le recall (≥ 98 %), ni pour le FPR (0,14 % contre < 0,1 %).
Sur l'ensemble des sessions, 1 597 reçoivent le verdict `ALERTE`, 6 122 `NON_EVALUE` et 177 260 `NORMAL`.

### 3.3 Par famille (test)

| Groupe | Famille C2 | Sessions | Détectées |
|---|---|---|---|
| web | **trickbot** | 1 469 | **781 (53,2 %)** |
| web | dridex, neris, virut, rbot, dns2tcp, iodine, cobaltstrike | 600, 184, 37, 12, 19, 20, 5 | **0** |
| dns | exfiltration DNS | 7 | 7 (100 %) |
| dns | neris, virut, rbot (CTU-13) | 1 251, 541, 12 | 22 (1,8 %), 38 (7,0 %), 0 |

| Famille bénigne | Sessions | Fausses alertes |
|---|---|---|
| web : benign_https / benign_doh / benign_normal_user | 75 735 / 999 / 857 | 105 (0,14 %) / 0 / 0 |
| dns : benign_dns / benign_normal_user | 30 045 / 267 | 15 (0,05 %) / 29 (10,9 %) |

### 3.4 Familles non vues (`unseen`, 221 sessions dont 143 `unseen_strict`)
DNS : 19 / 68 (27,9 %), toutes sessions DNS « banales » de CTU-13 labellisées C2 par convention. Web : 0 / 153 (0 / 75 `unseen_strict`). Total : 19 / 221 (8,6 %).

## 4. Analyse

1. **La fusion réduit les fausses alertes à rappel égal ou voisin.** À FPR cible identique, la règle « et » ramène le FPR du web de 0,29 % (LSTM seul) et 0,32 % (Transformer seul) à 0,14 %, pour un rappel de 41,7 % contre 42,2 % et 46,5 %. Par rapport à l'autoencodeur seul, le FPR est divisé par 30 (4,26 % → 0,14 %), au prix de 28 points de rappel (69,9 % → 41,7 %).
2. **Le LSTM n'ajoute rien à la fusion** : `ae+transformer+lstm` en « et » est identique à `ae+transformer` en « et » (0,14 % / 41,7 %). Les scores de l'autoencodeur et du Transformer sont peu corrélés sur le bénin (corrélation de rang 0,15), ce qui explique l'efficacité de la règle « et ».
3. **Le rappel est porté par une seule famille** : tout ce que la fusion détecte dans le web est du trickbot ; le dridex, que l'autoencodeur seul détectait à 40 %, est perdu parce que le Transformer ne le signale pas. Les sessions de trickbot sont très répétitives (voir P3) : le rappel est probablement optimiste.
4. **La sélection automatique est inutilisable** : elle choisit `max` (recall de validation de 84,8 %) mais le FPR de test est de 3,4 %, 30 fois la cible : un seuil calibré sur 2 170 bénins ne se transfère pas.
5. **Choix de la règle « et »** : justifié par la priorité donnée au FPR et par la faible corrélation entre détecteurs. **Les résultats du test de tous les candidats ont été vus avant ce choix** : il n'est donc pas strictement indépendant du test.
6. **Sessions web sans séquence** : le modèle séquentiel ne peut pas les juger. Avec l'autoencodeur seul (option par défaut `--noseq ae`), calibré sur 98 bénins de validation, elles produisent 372 fausses alertes (FPR de 15 % sur 2 488 bénins) pour 127 détections (26,9 % de 472 C2). L'**abstention** (`--noseq abstain`) retire ces fausses alertes : ces sessions ne déclenchent aucune alerte (verdict `NON_EVALUE`) et leurs 472 C2 sont comptés comme manqués.
7. **DNS** : seul l'autoencodeur est utilisable. Le rappel global (3,7 %) est dominé par les sessions DNS banales de CTU-13 ; la vraie exfiltration est détectée à 7 sur 7, mais avec un effectif minuscule (borne basse 65 %).

## 5. Limites

- Validation web de 2 170 bénins (2 fausses alertes autorisées) et DNS de 9 689 (9) : seuils **peu fiables**, d'où l'écart entre FPR de validation (0,09 %) et de test.
- La strate `web_noseq` n'a que 98 bénins de validation : elle ne peut pas être calibrée.
- Découpage officiel de P2 : bénin web concentré dans le test, mêmes machines entre ensembles (voir `docs/p5_6b_autoencoder_train.md` et `docs/demande_P2_rebalancement.md`).
- Les trois modèles ont été entraînés séparément, avec leurs propres choix ; une seule graine ; le re-split du bénin web n'a pas été appliqué ici (il exigerait de réentraîner le Transformer et le LSTM).
- Rappel par famille parfois sur de très petits effectifs.
- Le recall de validation sert à choisir la règle, mais la validation web est dominée par le trickbot (84 % des C2 de validation).

## 6. Reproduire

```bash
# 1. scores du LSTM (depuis le code de la branche feature/wassima-sprint3 ; vérifie 1 874 C2 / 75 103 bénins / 42,16 % / 0,29 %)
python -I p6_models/scripts/p6_07a_export_lstm_scores.py --p5 <partage_wassima>/p5_features_tls_seq/output --p2 <dossier sessions_pcap> \
   --model <partage_wassima>/p6_models/models/lstm_web_anomaly_packets.pt --out lstm_scores.parquet
# 2. fusion (scénario retenu)
python -I p6_models/scripts/p6_07_fusion.py --ae p6_models/output/ae_scores.parquet \
   --seq "transformer=<...>/transformer_scores_packets.parquet:transformer_score" --seq "lstm=<...>/lstm_scores.parquet:lstm_score" \
   --select "ae+transformer|and" --noseq abstain
```

Sorties : `fusion_metrics.csv` (15 candidats × strates), `fusion_by_family.csv`, `fusion_rule.json` (versionnés) ; `scores.parquet` et `fusion_reference.joblib` (non versionnés).

## 7. Utilisation dans le pipeline (P7)

```python
import joblib, importlib
fus = importlib.import_module("p6_07_fusion")
ref = joblib.load("p6_models/models/fusion_reference.joblib")     # seuils + références de rang
out = fus.apply_fusion(df, ref)       # df : colonnes group, ae_score, transformer_score (vide si pas de séquence)
# out["fused_score"], out["fused_alert"], out["stratum"], out["verdict"] (ALERTE / NORMAL / NON_EVALUE)
```
Le pipeline doit calculer `ae_score` (`score_autoencoder`) et `transformer_score` pour chaque session web avec séquence ; sans séquence valide, la session est `NON_EVALUE`.

## 8. Suite

1. Obtenir de P2 un découpage rééquilibré, puis recalibrer et réentraîner tous les modèles sur ce découpage.
2. Remplacer l'abstention par une vraie solution pour les sessions sans paquets (features de volume et de TLS).
3. Tableau de métriques finales (P6.7, Hajar) à partir de `fusion_metrics.csv`.
