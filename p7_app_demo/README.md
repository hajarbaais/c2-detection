# P7 — Démo live & explicabilité

## Contenu
- `app/` — application **Streamlit** : ingestion live → session → score → alerte (< 5 s)
- `shap_explanations.py` — explications SHAP pour les sessions alertées (P7.2)
- `docker/` — Dockerfile + docker-compose pour déployer la démo

## Pipeline bout-en-bout (P7.1) — `pipeline.py`

Squelette du sprint 2 : les interfaces entre lots sont figées, chaque étape est chronométrée.

```
extract(pcap) -> sessionize(logs) -> features(sessions) -> score(df) -> verdict(scores) -> alerts(...)
     P2              P2                 P4 (+P5)             P6         P6 fusion        P7.3 JSON
```

```bash
python -I p7_app_demo/pipeline.py --p2 <dossier output P2> --p5 <dossier p5_output> [--limit 5000]
```

| Étape | État |
|---|---|
| chargement des modèles une seule fois | ✅ règles calibrées + Isolation Forest + autoencodeur (`autoencoder.pt`) + référence TLS |
| `features` | ✅ P4 (`features_dns.parquet`) + P5 (`features_tls.parquet`, rareté/réputation recalculées avec `tls_reference.joblib`) — TODO : calcul en mémoire |
| `score` / `verdict` | ✅ règles, IF, AE et modèles séquentiels optionnels. Verdict actuel dès qu'un détecteur alerte |
| `alerts` | ✅ format provisoire — schéma JSON + MITRE sur la branche P7.3 de Salma |
| `shap` | ✅ jusqu'à trois contributions SHAP par session alertée par le LSTM, le Transformer, l'autoencodeur ou l'Isolation Forest |
| `extract` / `sessionize` sur un PCAP | ⏳ `NotImplementedError` — brancher les scripts P2 |

Un détecteur dont le fichier manque (ou dont les colonnes P5 manquent) est ignoré avec un message, sans erreur.

Mesure actuelle (184 979 sessions, hors extraction, CPU) : ~3,9 s au total dont 2,0 s de chargement des modèles
(une seule fois) ; sur les 184 979 sessions : 155 alertes règles, 539 IF, 5 826 AE (6 087 au total, 2 597 vrais C2).

## Démo
Montrer un cas **normal** et un cas **suspect**, avec l'explication SHAP de la décision.

## Explications SHAP (P7.2)

Le pipeline remplit `top_features` après la construction des alertes. Chaque entrée
contient `name` (colonne P4/P5 ou P2), `value` (valeur d'origine, `null` si manquante)
et `impact` (contribution SHAP signée). Jusqu'à trois contributions non nulles sont
affichées, par valeur absolue décroissante. Un impact positif augmente le score expliqué
par rapport à la référence bénigne. La somme des trois impacts affichés peut différer
de l'écart total au score de référence.

Pour une alerte déclenchée par plusieurs modèles, la liste explique le premier modèle
explicable dans l'ordre LSTM, Transformer, autoencodeur, Isolation Forest. Les impacts
sont exprimés dans l'unité du score du modèle choisi. Une alerte déclenchée uniquement
par les règles garde `top_features: []`.
Pour l'autoencodeur et l'Isolation Forest, les contributions sont estimées avec SHAP par
permutation, avec trois allers-retours complets
et jusqu'à 16 sessions bénignes d'entraînement du même groupe (DNS ou web). Le score
est recalculé avec le prétraitement habituel du détecteur. Les valeurs sont locales à
la session et au score choisi.

En mode hors ligne, la référence vient des sorties P2/P4/P5, y compris en présence de
`--limit` ou de `--session-id` (option répétable). Exemple pour rejouer une session, avec
les fichiers fournis par l'équipe placés dans les dossiers du dépôt :

```powershell
python -I p7_app_demo/pipeline.py --p2 data/P2_livrables_v3/output --session-id <identifiant>
```

Pour un appel avec des sessions sans labels, fournir à `C2Pipeline`
un `shap_reference` déjà construit par `features()` à partir d'un jeu de référence
contenant `group`, `split` et `label`. Sans référence bénigne pour le groupe, l'alerte
reste disponible avec `top_features: []` et le pipeline émet un avertissement.
Le chronométrage `shap` inclut la préparation de la référence et le calcul des contributions.

Pour rejouer le lot d'équipe, utiliser le `autoencoder.pt` standard d'Oumaima et
les deux `iforest_*.joblib` de Wassima avec leur `tls_reference.joblib`, dans
`p6_models/models/`. Les fichiers Isolation Forest transmis ont été entraînés
avec `scikit-learn==1.9.1`. Les poids et les fichiers Parquet restent hors Git.

### LSTM et Transformer

Activer un modèle séquentiel avec `--lstm` ou `--transformer` et le chemin de son
checkpoint. Le dossier `--p5` doit contenir le `features_seq.npz` P5 correspondant.
Le pipeline joint les séquences par `session_id`, reprend le score et le seuil P6,
et respecte le sous-ensemble `packets` du checkpoint web et sa longueur d'entrée.

```powershell
python -I p7_app_demo/pipeline.py --p2 data/P2_livrables_v3/output --p5 p5_features_tls_seq/output --lstm p6_models/models/lstm_web_anomaly_packets.pt --transformer p6_models/models/tf_web_anomaly_packets.pt --limit 100
```

SHAP exact traite chaque canal P5 comme une contribution sur l'ensemble des pas
valides. La référence remplace un canal par la médiane des pas bénins d'entraînement
du même groupe, avec le masque et le padding de la session conservés. Pour une
entrée `seq_*`, `value` est la moyenne du canal sur les pas valides dans l'échelle
P5. Une séquence vide ou sans référence bénigne peut utiliser l'explication d'un
autre détecteur qui a déclenché l'alerte.
