# P6.5 : Transformer sur les séquences P5 (Imane)

Modèle séquentiel du sprint 3, entraîné sur `features_seq.npz` (séquences de 64 pas × 6 features produites par P5 : `log_dt`, `log_size`, `log_resp_size`, `direction`, `is_dns`, `is_nxdomain`).

## Fichier

`scripts/p6_05_transformer.py` : même conventions que `p5_6b_autoencoder_train.py` (un modèle par groupe dns / web, seed 42, seuil fixé sur le **bénin de validation à 0,1 % de FPR**, le test n'est jamais utilisé pour choisir).

| Option | Rôle |
|---|---|
| `--mode supervised` | classifieur C2 / bénin (BCE avec poids de classe, score = logit) |
| `--mode anomaly` | auto-encodeur de séquences entraîné sur le bénin du train seul (score = erreur de reconstruction) |
| `--subset packets` | web : écarte les sessions « fallback » (aucun paquet réel : `direction = 0` et `is_dns = 0`) ; train, val, test et seuil sont recalculés |
| `--max-len N` | tronque les séquences aux N premiers pas (16 pour dns : ~2 pas valides en moyenne) |
| `--p2 <dossier>` | ajoute les métriques par famille et `unseen_strict` |
| `--out-name` | nom du `.pt` (gitignoré, à partager par Drive) |

```
python -I p6_models/scripts/p6_05_transformer.py --p5 p5_features_tls_seq/output --p2 <dossier output P2> \
       --mode anomaly --groups web --subset packets --no-mlflow --out-name tf_web_anomaly_packets
python -I p6_models/scripts/p6_05_transformer.py --p5 p5_features_tls_seq/output \
       --mode anomaly --groups dns --max-len 16 --no-mlflow --out-name tf_dns_anomaly
```

Architecture : encodeur Transformer (2 couches, d_model 64, 4 têtes) avec masque de padding ; moyenne sur les pas valides ; tête linéaire (supervised) ou goulot z = 16 puis décodeur (anomaly). Entraînement sur CPU (web ≈ 20 min, dns ≈ 13 min).

## Résultats (seuil à 0,1 % de FPR sur le bénin de validation)

| Essai | Split | Rappel test | FPR test | ROC-AUC test |
|---|---|---|---|---|
| web, supervised, tout | actuel | 0,5 % | 0,01 % | 0,68 |
| web, anomaly, tout | actuel | 42,2 % | 0,77 % | 0,79 |
| web, anomaly, sessions avec paquets | actuel | 46,5 % | 0,32 % | 0,90 |
| dns, anomaly, `--max-len 16` | actuel | 0,11 % | 0,06 % | 0,88 |

CPU : environ 1 à 2 ms par session (lot de 1), 0,35 à 0,51 Mo, 84 016 paramètres (dns) et 124 240 (web).
Rappel sur `unseen` : 0,65 % (web, tout), 0 % (web, avec paquets : 0 sur 151, dont 0 sur 73 `unseen_strict`), 0 % (dns : 0 sur 68).

Par famille (web, anomaly, sessions avec paquets, test) :

| Famille | Sessions C2 | Rappel |
|---|---|---|
| trickbot | 1 145 | 75,2 % |
| neris | 168 | 6,0 % |
| dridex | 480 | 0 % |
| virut | 34 | 0 % |
| iodine | 20 | 0 % |
| dns2tcp | 19 | 0 % |
| cobaltstrike | 5 | 0 % |
| rbot | 3 | 0 % |

Fausses alertes par type de bénin : `benign_https` 0,32 % (234 sur 73 603), `benign_normal_user` 0,88 % (7 sur 791, petit effectif), `benign_doh` 0 % (0 sur 709).

## Lecture

- **Le supervisé ne généralise pas** : la perte de validation explose pendant que celle du train baisse. Le mode anomaly, qui ne voit que du bénin, ne peut pas apprendre la capture d'origine et donne de bien meilleurs résultats sur le même split.
- **Pas de raccourci « fallback »** : en retirant les sessions fallback (44 % des séquences C2 web contre 3,6 % des bénignes), les scores montent. Le modèle séquentiel n'a presque rien à lire dans ces sessions. Les deux colonnes ne portent pas sur le même ensemble : ne pas écrire « le modèle passe de 0,79 à 0,90 ».
- **Le rappel global web est presque entièrement trickbot** (861 des 871 détections). La moyenne par famille est d'environ 10 %. Les règles et l'Isolation Forest détectent 0 % du trickbot : c'est la première preuve que les séquences apportent quelque chose.
- **DNS** : le classement est correct (AUC 0,88) mais le rappel à 0,1 % de FPR est quasi nul. Les séquences DNS sont aussi courtes que les bénignes ; l'information utile (entropie du nom de domaine) est dans les features P4, pas dans les séquences. Utiliser le Transformer comme entrée de la fusion, pas seul.

## Limites

- Le seuil web repose sur 2 268 bénins en validation (0,1 % = 2 alertes) : la FPR test est 3 à 8 fois plus haute que celle de validation. À refaire avec le découpage rééquilibré de P2 (`docs/demande_P2_rebalancement.md`).
- Aucune détection des familles inconnues.
- Dridex (480 sessions) n'est pas détecté : cause non encore analysée.
- Les fichiers `evaluation/transformer_<mode>_<subset>_*.csv` sont nommés par mode et sous-ensemble : un nouvel essai avec le même couple les écrase (le CSV `anomaly_all` contient aujourd'hui le dns). Les chiffres ci-dessus viennent des journaux d'exécution.
- `--max-len` s'applique à tous les groupes d'un même essai : un `.pt` unique dns + web demande une valeur par groupe (à faire).

## Interface pour la fusion et le pipeline

```python
from p6_05_transformer import load_transformer, score_transformer
loaded = load_transformer("p6_models/models/tf_web_anomaly_packets.pt")
df = score_transformer(loaded, X, mask, groups)   # colonnes transformer_score, transformer_alert
```

`transformer_score` : plus il est élevé, plus la session est suspecte (erreur de reconstruction en mode anomaly, logit en mode supervised).

## À faire

1. Relancer tous les essais avec le split rééquilibré de P2.
2. Aligner la méthode (supervisé ou anomalie) avec le LSTM de Wassima, puis remplir le tableau comparatif.
3. Analyser dridex, puis évaluer le Transformer dans la fusion d'Oumaima.
4. Tests Red Team (P7) : perturber timing et tailles, réentraîner, mesurer la robustesse.
