# P7 — Démo live & explicabilité

## Contenu
- `app/` — application **Streamlit** : ingestion live → session → score → alerte (< 5 s)
- `shap/` — explications SHAP par prédiction
- `docker/` — Dockerfile + docker-compose pour déployer la démo

## Pipeline bout-en-bout (P7.1) — `pipeline.py`

Squelette du sprint 2 : les interfaces entre lots sont figées, chaque étape est chronométrée.

```
extract(pcap) -> sessionize(logs) -> features(sessions) -> score(df) -> verdict(scores) -> alerts(...)
     P2              P2                 P4 (+P5)             P6         P6 fusion        P7.3 JSON
```

```bash
python -I p7_app_demo/pipeline.py --p2 <dossier output P2> [--limit 5000]
```

| Étape | État |
|---|---|
| chargement des modèles une seule fois | ✅ règles calibrées + Isolation Forest |
| `features` | ✅ relit `features_dns.parquet` (jointure gauche) — TODO : calcul en mémoire, + P5 |
| `score` / `verdict` | ✅ règles + IF, verdict = au moins un détecteur — TODO S3 : AE, LSTM, fusion |
| `alerts` | ✅ format provisoire — TODO S3 : schéma JSON + MITRE (Salma), SHAP (Farah) |
| `extract` / `sessionize` sur un PCAP | ⏳ `NotImplementedError` — brancher les scripts P2 |

Mesure actuelle (184 979 sessions, hors extraction) : ~3 s au total, ~0,3 s pour 2 000 sessions.

## Démo
Montrer un cas **normal** et un cas **suspect**, avec l'explication SHAP de la décision.
