# P5.6a — Prototype d'autoencodeur (Oumaima)

- **Tâche** : P5.6a, Sprint 2 — « Architecture PyTorch testée sur échantillon »
- **Script** : `p6_models/scripts/p5_6a_autoencoder_proto.py`
- **Date** : 12/11/2026 — **Branche** : `feature/oumaima-sprint2`
- **Statut** : prototype validé (la chaîne complète tourne). **Ce n'est pas le modèle final** (voir P5.6b).

## 1. Objectif

Vérifier, sur un échantillon du trafic bénin, que la chaîne de l'autoencodeur fonctionne de bout en
bout : prétraitement, architecture PyTorch, entraînement, seuil de détection, évaluation. Les règles
d'évaluation sont celles de P6 (Wassima), afin de comparer l'autoencodeur à l'Isolation Forest.

## 2. Principe

Un autoencodeur est un réseau de neurones qui apprend à reconstruire ses entrées. Il est entraîné
**uniquement sur du trafic bénin**. Une session qui s'éloigne de ce trafic est mal reconstruite : son
erreur de reconstruction (MSE) est élevée et sert de **score d'anomalie**.

Une anomalie n'est pas une preuve d'attaque : un service légitime peut aussi être mal reconstruit.

## 3. Données

| Fichier                 | Origine    | Utilisation                                                  |
| ----------------------- | ---------- | ------------------------------------------------------------ |
| `sessions_pcap.parquet` | P2         | labels, split, type de session, volumes                      |
| `features_dns.parquet`  | P4 (Hajar) | features DNS et temporelles                                  |
| `features_tls.parquet`  | P5 (Imane) | 14 features TLS/HTTPS, groupe web uniquement (option `--p5`) |

Jointure sur `session_id`. Les colonnes P5 sont préfixées `p5_` : `sessions_pcap` contient déjà
`tls_version` et `cert_self_signed` (valeurs brutes), elles auraient été écrasées.

**Séparation des sessions.** Chaque session porte un type donné par P2. Deux groupes, un modèle par groupe :
`dns` (sessions DNS) et `web` (sessions HTTPS et HTTP). Les features ne sont pas les mêmes.

| Ensemble                                                 | Utilisation                                                       |
| -------------------------------------------------------- | ----------------------------------------------------------------- |
| train, bénin seulement, 20 % tirés au hasard (graine 42) | entraînement et prétraitement                                     |
| val, bénin seulement                                     | arrêt anticipé et choix du seuil                                  |
| test                                                     | mesure finale                                                     |
| unseen / unseen_strict                                   | généralisation à des familles non vues (seul le rappel a un sens) |

## 4. Méthode

**Prétraitement** (appris sur le bénin du train uniquement) : `log1p` sur les colonnes positives et très
asymétriques, indicateurs de valeur manquante, remplacement par la médiane, standardisation, écrêtage à ±10.

**Architecture** : MLP symétrique `d → 64 → 32 → 8 → 32 → 64 → d` (ReLU). Perte MSE, optimiseur Adam
(lr 1e-3), lots de 256, 30 époques maximum, arrêt anticipé (patience 5) sur la perte du bénin de validation.

**Seuil** : choisi sur le bénin de **validation** pour un FPR de 0,1 %. Le test n'est jamais utilisé pour
choisir quoi que ce soit.

**Métriques** : `p6_models/evaluation/metrics.py` (rappel, FPR, ROC-AUC, intervalles de Wilson), globalement et par famille.

## 5. Résultats

| Groupe | Colonnes | Entrée | Paramètres | Sessions train / val | Perte train   | Perte val     |
| ------ | -------- | ------ | ---------- | -------------------- | ------------- | ------------- |
| dns    | 24       | 29     | 8 549      | 9 464 / 9 689        | 0,653 → 0,020 | 0,402 → 0,019 |
| web    | 25       | 36     | 9 452      | 1 767 / 2 268        | 0,699 → 0,127 | 1,248 → 0,289 |

| Groupe | Split         | C2    | Bénin  | Rappel | FPR    | ROC-AUC                              |
| ------ | ------------- | ----- | ------ | ------ | ------ | ------------------------------------ |
| dns    | val           | 142   | 9 689  | 10,6 % | 0,09 % | 0,869                                |
| dns    | test          | 1 811 | 30 312 | 3,4 %  | 0,12 % | **0,929** (Isolation Forest : 0,804) |
| dns    | unseen        | 68    | –      | 23,5 % | –      | –                                    |
| web    | val           | 1 102 | 2 268  | 2,7 %  | 0,09 % | 0,957                                |
| web    | test          | 2 346 | 77 591 | 2,2 %  | 1,81 % | 0,803 (Isolation Forest : 0,924)     |
| web    | unseen        | 153   | –      | 1,3 %  | –      | –                                    |
| web    | unseen_strict | 75    | –      | 2,7 %  | –      | –                                    |

Par famille, sur le test :

- **dns** : exfiltration DNS-EXF 100 % (deux familles) ; neris 1,6 %, rbot 0 %, virut 6,3 % ; FPR bénin_dns 0,05 %, benign_normal_user 7,9 %.
- **web** : cobaltstrike 60 % (seulement 5 sessions) ; dns2tcp, dridex, iodine, neris, rbot 0 % ; trickbot 3,2 %, virut 2,7 % ; FPR benign_doh **36,5 %**, benign_https 1,4 %.

## 6. Analyse

- **DNS** : le classement de l'autoencodeur est meilleur que celui de l'Isolation Forest (ROC-AUC 0,93 contre 0,80). Mais au seuil de 0,1 % de FPR, il ne détecte presque que l'exfiltration. Le DNS de CTU-13 (labellisé C2 par convention, ce sont les requêtes de la machine infectée) ressemble au DNS normal.
- **Web** : moins bon que l'Isolation Forest. Le bénin DoH, d'une autre distribution que le reste du bénin d'entraînement, est mal reconstruit (36,5 % de fausses alertes), ce qui explique le FPR de 1,81 %.
- **Objectifs du prof** (rappel ≥ 98 %, FPR < 0,1 %) : non atteints. Les chiffres ci-dessus sont ceux d'un prototype et ne sont pas des résultats finaux.

## 7. Limites

1. Entraînement sur 20 % du bénin seulement (1 767 sessions pour le web).
2. Seuil web peu fiable : 0,1 % de 2 268 sessions de validation = 2,3 faux positifs autorisés.
3. Une seule exécution (graine 42), sans répétition ni intervalle de variation.
4. Comparaison avec l'Isolation Forest : celle-ci a été entraînée sur tout le bénin du train, pas sur 20 %.
5. Les machines sont partagées entre train et test : le test reste optimiste par rapport à un vrai déploiement.
6. Les features P5 `ja3_rarity` et `ja4_rarity` sont calculées sur tout le train, C2 compris (à recalculer sur le bénin seul).
7. Les colonnes de P3 (`ctu13_broad_label`, `share_no_payload`…) ne sont pas utilisées : le chargement lit `sessions_pcap`, pas `sessions_unified`.
8. Effectifs faibles pour certaines familles (par exemple cobaltstrike : 5 sessions).

## 8. Reproduire

```bash
git switch -c feature/oumaima-sprint2 origin/feature/wassima-sprint2
pip install torch pandas numpy pyarrow
# placer features_dns.parquet dans p4_features_dns_time/output/
python -I p6_models/scripts/p5_6a_autoencoder_proto.py \
  --p2 <dossier contenant sessions_pcap.parquet> --p5 <dossier p5_output>
```

Les données ne sont pas versionnées (trop volumineuses). Sorties : `p6_models/output/` (gitignoré).
Version de PyTorch utilisée : 2.15.0+cpu.

## 9. Suite (P5.6b)

1. Entraînement sur tout le bénin du train (`--sample-frac 1`).
2. Suivi MLflow (expérience `c2-autoencoder`, tags `lot: P5`, `task: P5.6`), courbes de perte.
3. Sauvegarde de `p6_models/models/autoencoder.pt` (ajouter `*.pt` au `.gitignore`).
4. Recalculer `ja3_rarity` / `ja4_rarity` sur le bénin du train, et traiter le bénin DoH.
5. Tester avec et sans `ctu13_broad_label` (README P6).
