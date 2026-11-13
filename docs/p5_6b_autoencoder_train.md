# P5.6b — Entraînement de l'autoencodeur sur le trafic bénin (Oumaima)

- **Tâche** : P5.6b, Sprint 2 — « Entraînement complet suivi dans MLflow, `autoencoder.pt` »
- **Script** : `p6_models/scripts/p5_6b_autoencoder_train.py` (réutilise `p5_6a_autoencoder_proto.py`)
- **Branche** : `feature/oumaima-sprint2` — **Date** : 13/11/2026
- **Runs MLflow** : expérience `c2-autoencoder`, runs `ae_v1` et `ae_v2` (version retenue : `ae_v2`)

| Critère de la tâche | État |
|---|---|
| Entraînement sur bénin seul | Fait (train bénin : 47 318 sessions DNS, 8 836 web) |
| `autoencoder.pt` | Fait : `p6_models/models/autoencoder.pt` (dns + web dans un seul fichier, non versionné) |
| Courbes dans MLflow | Fait : `train/<groupe>/loss` et `val/<groupe>/loss`, par époque |

## 1. Ce qui change par rapport au prototype P5.6a

| | P5.6a | P5.6b |
|---|---|---|
| Bénin du train utilisé | 20 % | 100 % |
| Époques | 30 max, patience 5 | `ae_v1` : 100 / 10 ; `ae_v2` : 300 / 20 |
| Features TLS `ja3_rarity`, `ja4_rarity`, `log_domain_reputation` | calculées par P5 sur tout le train (C2 compris) | **recalculées sur le bénin du train web seul** |
| Suivi | aucun | MLflow (paramètres, courbes, métriques, artefacts) |
| Sauvegarde | `p6_models/output/` (ignoré) | `p6_models/models/autoencoder.pt` |
| Évaluation DNS | globale | globale **et sans les requêtes DNS banales de CTU-13** (`--p3`) |
| Intégration pipeline | non | `load_autoencoder()` et `score_autoencoder()` |

Rareté recalculée : sur le test web, la moyenne de `ja3_rarity` du C2 passe de 2,57 (version P5, où les
empreintes des C2 du train les faisaient paraître courantes) à 6,50 (bénin seul), tandis que le bénin
reste à ~2,6.

## 2. Données et séparation

Fichiers : `sessions_pcap.parquet` (P2), `features_dns.parquet` (P4), `features_tls.parquet` (P5, groupe web),
`sessions_unified.parquet` (P3, option `--p3`, **pour l'évaluation uniquement** : `ctu13_broad_label`).

Deux groupes, un modèle par groupe : `dns` (sessions DNS) et `web` (HTTPS + HTTP). Le type de session est déjà
fourni par P2. Features : 24 colonnes (dns) et 25 colonnes (web), listées dans les artefacts MLflow
`features_dns.txt` et `features_web.txt`.

| Ensemble | Utilisation |
|---|---|
| train, bénin | entraînement et ajustement du prétraitement |
| val, bénin | arrêt anticipé et choix du seuil |
| test | mesure finale |
| unseen / unseen_strict | familles non vues (rappel seulement) |

## 3. Méthode

- **Prétraitement** (appris sur le bénin du train) : `log1p` sur les colonnes positives très asymétriques,
  indicateurs de valeurs manquantes, médiane, standardisation, écrêtage à ±10.
- **Architecture** : MLP symétrique `d → 64 → 32 → 8 → 32 → 64 → d` (ReLU), perte MSE.
- **Entraînement** : Adam, lr 1e-3, lots de 256, graine 42, arrêt anticipé sur la perte du bénin de validation.
- **Score** : erreur de reconstruction (MSE) par session ; score élevé = suspect.
- **Seuil** : bénin de validation, FPR cible 0,1 %. Le test n'intervient jamais dans un choix.
- **Métriques** : `p6_models/evaluation/metrics.py` (rappel, FPR, précision, ROC-AUC, PR-AUC, intervalles de Wilson à 95 %).

## 4. Suivi MLflow

- Expérience `c2-autoencoder`, tags `lot: P5`, `task: P5.6`, `author: oumaima`, `git_commit`, `features`.
- Paramètres (19) : architecture, lr, lot, époques, patience, `sample_frac`, `target_fpr`, `threshold_split`, graine, tailles d'entrée.
- Métriques (68 pour `ae_v2`) nommées `<split>/<groupe>/<métrique>`, plus `train/<groupe>/loss` et `val/<groupe>/loss` par époque.
- Artefacts : `models/autoencoder.pt`, `autoencoder_metrics.csv`, `features_dns.txt`, `features_web.txt`.
- Consultation : `mlflow ui --backend-store-uri sqlite:///mlflow.db`, onglet **Model training**, run, **Model metrics**.
  La base `mlflow.db` et le dossier `mlruns/` sont locaux (gitignorés).

## 5. Résultats

### 5.1 Entraînement

| Run | Groupe | Époques | Durée | Meilleure perte val |
|---|---|---|---|---|
| ae_v1 | dns | 76 | 35 s | 0,0044 (époque 66) |
| ae_v1 | web | 100 | 8 s | 0,0723 (époque 98) |
| ae_v2 | dns | 123 | 138 s | 0,0040 (époque 103) |
| ae_v2 | web | 281 | 57 s | 0,0754 (époque 261) |

### 5.2 Comparaison des deux runs (test)

| | ae_v1 | ae_v2 | Isolation Forest (Wassima) |
|---|---|---|---|
| DNS : rappel / FPR | 3,9 % / 0,15 % | 3,7 % / 0,15 % | – |
| DNS : ROC-AUC | 0,907 | **0,935** | 0,804 |
| Web : rappel / FPR | 55,8 % / 5,03 % | 55,8 % / **4,13 %** | – |
| Web : ROC-AUC | 0,898 | 0,908 | **0,924** |

### 5.3 Détail de `ae_v2` (seuil à 0,1 % de FPR sur le bénin de validation)

| Groupe | Split | C2 | Bénin | Rappel [IC 95 %] | FPR | Précision | ROC-AUC |
|---|---|---|---|---|---|---|---|
| dns | val | 142 | 9 689 | 10,6 % [6,5 – 16,7] | 0,09 % | 62,5 % | 0,892 |
| dns | val, sans CTU-13 | 8 | 9 689 | 100 % [67,6 – 100] | 0,09 % | 47,1 % | 0,9999 |
| dns | test | 1 811 | 30 312 | 3,7 % [2,9 – 4,7] | 0,15 % | 60,4 % | 0,935 |
| dns | test, sans CTU-13 | 7 | 30 312 | **100 % [64,6 – 100]** | 0,15 % | 13,7 % | 0,9999 |
| dns | unseen | 68 | – | 27,9 % [18,7 – 39,6] | – | – | – |
| web | val | 1 102 | 2 268 | 62,6 % [59,7 – 65,4] | 0,09 % | 99,7 % | 0,976 |
| web | test | 2 346 | 77 591 | 55,8 % [53,8 – 57,8] | **4,13 %** [3,99 – 4,27] | 29,0 % | 0,908 |
| web | unseen | 153 | – | 0,65 % [0,1 – 3,6] | – | – | – |
| web | unseen_strict | 75 | – | 1,3 % [0,2 – 7,2] | – | – | – |

### 5.4 Par famille (test, `ae_v2`)

| Groupe | Famille C2 | Sessions | Rappel |
|---|---|---|---|
| dns | exfiltration DNS (2 familles) | 7 | 100 % |
| dns | neris / virut / rbot (CTU-13) | 1 251 / 541 / 12 | 1,8 % / 7,0 % / 0 % |
| web | trickbot | 1 469 | 72,5 % |
| web | dridex | 600 | 40,0 % |
| web | cobaltstrike (5 sessions) | 5 | 60,0 % |
| web | neris / virut / rbot / dns2tcp / iodine | 184 / 37 / 12 / 19 / 20 | 0,5 % / 2,7 % / 0 % / 0 % / 0 % |

| Groupe | Famille bénigne | Sessions | FPR |
|---|---|---|---|
| dns | benign_dns | 30 045 | 0,05 % |
| dns | benign_normal_user | 267 | 10,9 % |
| web | benign_https | 75 735 | 3,4 % (2 584 fausses alertes) |
| web | benign_doh | 999 | **61,6 %** |
| web | benign_normal_user | 857 | 0,35 % |

## 6. Analyse

**DNS.** Le classement de l'autoencodeur est meilleur que celui de l'Isolation Forest (0,935 contre 0,804). Au seuil
strict de 0,1 % de FPR, le rappel global est de 3,7 %, mais 1 804 des 1 811 sessions C2 du test sont les requêtes DNS
banales de la machine infectée de CTU-13, labellisées C2 par convention (colonne `ctu13_broad_label`). Sur la vraie
exfiltration DNS, le modèle détecte 7 sessions sur 7, mais avec un effectif minuscule (borne basse de l'IC à 95 % :
64,6 %), un FPR de 0,15 % (supérieur à la cible) et une précision de 13,7 % (44 fausses alertes).

**Web.** Le rappel de 55,8 % et le FPR de 4,13 % s'expliquent par la composition des ensembles (section 7) :
- 1 304 des 1 309 détections du test sont du trickbot ou du dridex ;
- le FPR est de 0,09 % en validation et de 4,13 % en test : le bénin du test (DoH et HTTPS de CIRA-CIC-DoHBrw) est
  très différent, en volume et en composition, du bénin du train.
Passer de 100 à 300 époques ne change presque rien (perte val 0,0723 → 0,0754) : le problème vient des données.

**Rappel sur `unseen`.** L'autoencodeur est entraîné sur du bénin seulement : il ne voit aucun C2, ni dans le train ni
ailleurs. Un faible rappel sur `unseen` signifie que ces familles ressemblent au bénin dans l'espace des features,
pas qu'elles n'ont pas été apprises.

**Objectifs du professeur** (rappel ≥ 98 %, FPR < 0,1 %) : **non atteints** sur le test, dans aucun des deux groupes.
Ces chiffres sont à présenter tels quels, avec les limites ci-dessous.

## 6 bis. Expérience : le déséquilibre du bénin web est-il la cause du FPR ? (run `ae_resplit`)

**Hypothèse** : le FPR web de 4,13 % en test (contre 0,09 % en validation) vient de la composition des ensembles
(bénin web : 10 % train, 2,6 % val, 87,5 % test ; `benign_doh` : 21 sessions en train).

**Protocole** (option `--resplit-benign-web`, ne modifie pas les fichiers de P2) : le bénin web est redécoupé dans chaque
couple (pcap, famille) par ordre des horodatages : 60 % train, 15 % val, 21 % test, avec un écart de 2 % entre ensembles.
Le DNS et les C2 ne changent pas. Même modèle, mêmes hyperparamètres (300 époques max, patience 20).

| Web | `ae_v2` (split de P2) | `ae_resplit` |
|---|---|---|
| Bénin train / val / test | 8 836 / 2 268 / 77 591 | 55 384 / 13 906 / 16 303 |
| FPR validation | 0,09 % | 0,09 % (13 / 13 906) |
| **FPR test** | **4,13 %** [3,99 – 4,27] | **0,56 %** [0,46 – 0,69] |
| FPR `benign_doh` | 61,6 % | 0 % (0 / 214) |
| FPR `benign_https` | 3,4 % | 0,57 % (90 / 15 909) |
| Rappel test | 55,8 % [53,8 – 57,8] | 59,6 % [57,6 – 61,5] |
| Rappel trickbot / dridex | 72,5 % / 40,0 % | 77,9 % / 41,2 % |
| Précision test | 29,0 % | 93,8 % (non comparable : le bénin du test passe de 77 591 à 16 303) |
| ROC-AUC test | 0,908 | 0,886 |

Le DNS et `unseen` sont inchangés (mêmes valeurs).

**Lecture**
- Le FPR web de test est divisé par 7 (4,13 % → 0,56 %) et les fausses alertes sur le DoH disparaissent : la composition des
  ensembles est la cause **principale** de l'écart. L'hypothèse est largement confirmée.
- Elle n'explique pas tout : 0,56 % reste 5,6 fois au-dessus de la cible de 0,1 %, et supérieur aux 0,09 % de la validation.
- Le rappel progresse peu (55,8 % → 59,6 %) : il vient toujours de trickbot et dridex. Les autres familles (rbot, neris,
  dns2tcp, iodine, virut, cobaltstrike) restent à 0–2 %. Un détecteur appris sur le bénin seul ne les distingue pas du
  bénin avec ces features : ce n'est pas un problème de données.
- **Borne optimiste** : mêmes machines, sessions voisines dans le temps entre train et test (le DoH bénin vient de 3 pcaps
  Firefox). Ce résultat montre ce qu'un bon découpage permet au mieux, pas la généralisation à de nouvelles captures.

## 7. Limites liées aux données (à traiter avec P2)

| Constat | Chiffres |
|---|---|
| Le bénin web est concentré dans le test | train 8 836 (10 %), val 2 268 (2,6 %), test 77 591 (87,5 %) |
| Le bénin web vient presque d'une seule source | 98 % de CIRA-CIC-DoHBrw (86 926 sessions), 2 % de Stratosphere-normal (1 769) |
| Le bénin DoH est quasi absent du train | `benign_doh` : 21 sessions en train, 16 en val, 999 en test (FPR 61,6 %) |
| Validation web trop petite pour calibrer 0,1 % | 2 268 × 0,1 % = 2,3 fausses alertes autorisées |
| Les familles C2 web sont réparties de façon inégale | rbot : 2 058 en train, 12 en test ; dridex : 0 en val ; trickbot : 236 / 921 / 1 469 (train / val / test) |
| Le rappel dépend de la composition de chaque split | web : 13 % sur le train, 62,6 % en val, 55,8 % en test |
| Labels DNS « par convention » | 1 804 des 1 811 C2 DNS du test ; 49 sessions d'exfiltration en tout |
| Machines partagées entre train et test | généralisation à de nouvelles machines non évaluée (voir P3) |

Autres limites : une seule graine (42), sans répétition ; seuils calibrés une seule fois ; `unseen_strict` ne contient
que 143 sessions (68 DNS, 75 web) ; les colonnes de P3 autres que `ctu13_broad_label` ne sont pas utilisées.
La demande détaillée adressée à P2 est dans `docs/demande_P2_rebalancement.md`.

## 8. Reproduire

```bash
git switch feature/oumaima-sprint2
pip install torch pandas numpy pyarrow mlflow
# placer features_dns.parquet dans p4_features_dns_time/output/
python -I p6_models/scripts/p5_6b_autoencoder_train.py \
  --p2 <dossier contenant sessions_pcap.parquet> \
  --p5 <dossier p5_output> \
  --p3 <chemin>/sessions_unified.parquet \
  --epochs 300 --patience 20 --run-name ae_v2
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Sorties : `p6_models/models/autoencoder.pt` (non versionné, à partager par Drive), `p6_models/evaluation/autoencoder_metrics.csv`,
`autoencoder_by_family.csv`. Version de PyTorch : `<à compléter>`.

## 9. Utilisation dans le pipeline (P7)

```python
from p5_6b_autoencoder_train import load_autoencoder, score_autoencoder
ae = load_autoencoder("p6_models/models/autoencoder.pt")   # une seule fois au démarrage
res = score_autoencoder(ae, df)                             # df avec colonne `group`, et features P4/P5
# res["ae_score"], res["ae_alert"] : à brancher dans C2Pipeline.score() (le verdict agrège les colonnes *_alert)
```
Le prétraitement et le seuil de chaque groupe sont dans le fichier `.pt`.

## 10. Suite

1. Obtenir de P2 un nouveau découpage (l'expérience `ae_resplit` montre un FPR web de test de 4,13 % → 0,56 %) (`docs/demande_P2_rebalancement.md`), puis relancer P5 (scaler, rareté) et ce script.
2. Rapporter le rappel **par famille** et en moyenne par famille, plus la version sans CTU-13 pour le DNS.
3. Le rappel web reste limité à trickbot et dridex : prévoir le modèle séquentiel et la fusion (sprint 3) pour les autres familles.
4. Brancher `ae_score` dans le pipeline avec Wassima.
