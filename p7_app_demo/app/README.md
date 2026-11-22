# P7 — Tableau de bord Streamlit (Aicha, Sprint 3)

Dépôt d'un PCAP → tableau des sessions (alertes en rouge) → détail + graphe SHAP → chronomètre.

| Fichier | Rôle |
|---|---|
| `app.py` | La page Streamlit (affichage uniquement, aucune détection recodée) |
| `dashboard_lib.py` | Fonctions sans Streamlit : mise en forme, contrôle du fichier déposé, données simulées |
| `../../tests/test_p7_dashboard.py` | 16 tests (pytest) |

## Lancer

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac : source .venv/bin/activate)
pip install streamlit pandas numpy pyarrow   # + scikit-learn joblib torch pour les vrais modèles
streamlit run p7_app_demo/app/app.py
```

La page s'ouvre sur <http://localhost:8501>.

## Les 3 sources de données

| Source | Ce qui est appelé | État |
|---|---|---|
| **Démo (données simulées)** | `dashboard_lib.demo_result()` | ✅ Marche sans données ni modèles. Bandeau « Données simulées » toujours affiché |
| **Sorties P2 déjà calculées** | `C2Pipeline.run_from_p2_output()` | ✅ Code prêt. Nécessite `sessions_pcap.parquet` (P2), `features_dns.parquet` (P4) et les modèles de `p6_models/models/` (Drive) |
| **Déposer un PCAP** | `C2Pipeline.run_pcap()` | ⏳ L'interface est prête, mais `extract` / `sessionize` lèvent encore `NotImplementedError` (P7.1). Un message clair s'affiche en attendant |

## Les 4 éléments de la tâche

| Élément | Comment |
|---|---|
| **Upload PCAP** | `st.file_uploader` (pcap, pcapng, cap ; 200 Mo max). Signature du fichier vérifiée. Le fichier est enregistré sous un **nom généré** (le nom donné par l'utilisateur n'est jamais réutilisé dans un chemin ou une commande), **jamais exécuté**, puis **supprimé** après l'analyse |
| **Tableau des sessions** | Une ligne par session, **alertes en rouge** et en tête. Filtres : alertes seulement, type, recherche. Cliquer une ligne ouvre le détail |
| **Détail + graphe SHAP** | Infos de la session, MITRE ATT&CK, alerte JSON (format P7.3). Graphe à partir de `top_features` (`name`, `value`, `impact`) : rouge = pousse vers C2, bleu = vers normal. Message « pas encore disponible » tant que `top_features` est vide |
| **Chronomètre** | `res["timings"]` du pipeline : une barre par étape, temps total comparé à l'objectif **< 5 s**. Le chargement des modèles (une seule fois, grâce à `st.cache_resource`) est affiché à part |

## Dépendances avec l'équipe

| Qui | Quoi | Effet sur le tableau de bord |
|---|---|---|
| Wassima / Hajar (P7.1) | Brancher `extract` et `sessionize` dans `pipeline.py` | Le mode « Déposer un PCAP » fonctionnera sans changer `app.py` |
| Farah (P7.2) | Remplir `top_features` avec SHAP | Le graphe s'affichera automatiquement |
| Salma (P7.3) | Merger `feature/p7.3-alert-json` | Sévérité et MITRE réels (le tableau de bord lit déjà ce format) |
| Oumaima (P7.6) | Ajouter `streamlit` (et `altair`) à `requirements-docker.txt` et activer le service `dashboard` de `docker-compose.yml` avec `streamlit run p7_app_demo/app/app.py` | Démo dans Docker |

## Sécurité

- Aucune vraie IP : seul `src_ip_hash` est affiché.
- Mode démo : domaines réservés `.example` / `.test` uniquement.
- Aucun fichier déposé ne reste sur le disque.
