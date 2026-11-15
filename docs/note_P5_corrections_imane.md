# Note pour Imane : corrections à apporter à P5 (features TLS et séquences)

- **De** : Wassima (P6, pipeline P7.1) — **Pour** : Imane (P5) — copie Oumaima (P5.6)
- **Date** : 15/11/2026 — **Branche vérifiée** : `p5-imane` (`2677ece`)
- **Données** : sorties P2 v3 (184 979 sessions) ; `p5_check.py` relancé : 0 NaN, séquences (184 979, 64, 6), aucune session vide.

Ton livrable fonctionne et respecte la règle principale (scalers ajustés sur le train). Les points ci-dessous
viennent de l'intégration dans P6 et dans le pipeline. Ils sont classés par priorité.

## 1. Rareté JA3/JA4 calculée avec les C2 du train (priorité 1)

`rarity_from_train()` compte les empreintes sur **tout** le train, C2 compris. Une empreinte de trickbot vue dans le train
paraît donc « courante » et le C2 du test ressemble au bénin.

| Rareté JA3 moyenne (web, test) | Version actuelle (tout le train) | Bénin du train seulement |
|---|---|---|
| Bénin | 2,72 | 2,63 |
| C2 | **2,57** (presque comme le bénin) | **6,50** |

**Correctif** : calculer les fréquences sur le **bénin du train web** (`split == "train"`, `label == 0`, sessions https/http).
C'est déjà fait côté P6 dans `p6_models/scripts/p6_tls.py` (même formule que l'autoencodeur d'Oumaima) :

```python
ref = (s["split"] == "train") & (s["label"] == 0)          # s = sessions https/http
f["ja3_rarity"] = rarity_from_train(s["ja3"], ref)          # au lieu de is_train
f["ja4_rarity"] = rarity_from_train(s["ja4"], ref)
```

## 2. Réputation de domaine : même problème, et une session se compte elle-même (priorité 1)

1. Comme la rareté, elle est calculée avec les C2 du train → même correctif (bénin du train seul).
2. Une session du train contribue à sa propre réputation : en train, elle n'est **jamais nulle**, alors qu'en test
   57 % des sessions bénignes ont une réputation de 0 (domaine jamais vu dans le train).

| Bénin web (référence = bénin du train) | train | val | test |
|---|---|---|---|
| `log_domain_reputation` moyen | 0,84 | 0,66 | 0,37 |
| part de valeurs nulles | **0 %** | 29 % | 57 % |

Le même effet existe pour la rareté JA3 : 0,95 en train contre 2,63 en test pour le bénin. Un modèle entraîné sur le
bénin du train voit alors le bénin du test comme « anormal », ce qui augmente les fausses alertes.

**Correctif proposé (à valider ensemble)** : pour les sessions de la référence, calculer la valeur **sans la session
elle-même** (« leave-one-out ») : retirer 1 au compte de son empreinte, et ne pas compter sa propre machine pour la
réputation si elle est la seule à contacter ce domaine.

Essai rapide (Isolation Forest web, split P2 actuel, non versionné) :

| | Référence actuelle | Leave-one-out |
|---|---|---|
| Rappel val / test | 2,3 % / 0,04 % | 36,7 % / 27,4 % |
| FPR val / test | 0,09 % / 0,15 % | 0,09 % / 0,50 % |
| ROC-AUC test | 0,949 | 0,948 |

Le rappel monte beaucoup, mais la FPR du test aussi : à refaire **après** le nouveau découpage de P2, avant de choisir.
Si on l'adopte, Oumaima et moi réentraînons l'autoencodeur et l'Isolation Forest (le pipeline doit recevoir exactement
les valeurs vues à l'entraînement).

## 3. Raccourcis possibles à documenter (priorité 2)

| Feature | Constat | Risque |
|---|---|---|
| `session_type == http` | 2 565 C2 contre 262 bénins (91 % de C2) | le modèle apprend « HTTP = C2 » |
| `tls_version == 0` | 66 % des C2 https (2 700 / 4 098) contre 7 % du bénin | reconnaît la capture, pas le comportement |
| `has_ja3` | 0,93 bénin contre 0,48 C2 | idem |

Le modèle supervisé de référence (HistGradientBoosting) en profite : à mentionner dans le README P5 et à tester au sprint 3
(modèle avec et sans ces colonnes).

## 4. Points techniques (priorité 3)

1. **Emplacement** : les scripts sont dans `p5/` à la racine. Les ranger dans `p5_features_tls_seq/scripts/` et écrire les
   sorties dans `p5_features_tls_seq/output/` (le pipeline lit ce dossier par défaut ; il est déjà dans le `.gitignore`).
2. **`tls_version_num`** : les motifs `"1.2"` sont des expressions régulières (le `.` accepte n'importe quel caractère).
   Ça marche sur les valeurs actuelles (`TLSv1.0`, `TLSv1.2`, `TLSv1.3`), mais une table exacte est plus sûre :
   `{"TLSv1.0": 1, "TLSv1.1": 2, "TLSv1.2": 3, "TLSv1.3": 4}`.
3. **`scaler.pkl`** : à réajuster après les correctifs 1 et 2 (les colonnes changent).
4. **Séquences** : vérifié, `seq` est déjà trié dans `events` et `events_pkts`, donc `groupby(...).head(L)` garde bien les
   L premiers pas. Ajouter un `sort_values(["session_id", "seq"])` avant la troncature protège si P2 change l'ordre.

## 5. Pour le pipeline en direct (P7, sprint 3)

Le pipeline relit aujourd'hui `features_tls.parquet`. Pour scorer une nouvelle capture, il faudra une fonction :

```python
def compute_tls(sessions: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """1 ligne par session https/http, colonnes = les 14 features, sans relire le train."""
```

où `reference` contient les tables de fréquence apprises sur le bénin du train (format de
`p6_tls.fit_tls_reference()`, sauvegardé dans `p6_models/models/tls_reference.joblib`). Tu peux réutiliser ce module.

## 6. À relancer après la livraison de P2 (`docs/demande_P2_rebalancement.md`)

`p5_features.py` (rareté, réputation et scaler dépendent du train), puis P6 (`p6_02_isolation_forest.py --p5 ...`) et
l'autoencodeur d'Oumaima.
