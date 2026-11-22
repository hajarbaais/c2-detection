# P7 — Démo live & explicabilité

## Contenu
- `app/` — tableau de bord **Streamlit** : dépôt PCAP, sessions (alertes en rouge), détail + SHAP, chronomètre → voir [`app/README.md`](app/README.md)
  Lancement : `streamlit run p7_app_demo/app/app.py`
- `shap/` — explications SHAP par prédiction
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
| `score` / `verdict` | ✅ règles + IF + AE (`ae_score`, `ae_alert`), verdict = au moins un détecteur — TODO S3 : LSTM, fusion |
| `alerts` | ✅ format provisoire — TODO S3 : schéma JSON + MITRE (Salma), SHAP (Farah) |
| `extract` / `sessionize` sur un PCAP | ⏳ `NotImplementedError` — brancher les scripts P2 |

Un détecteur dont le fichier manque (ou dont les colonnes P5 manquent) est ignoré avec un message, sans erreur.

Mesure actuelle (184 979 sessions, hors extraction, CPU) : ~3,9 s au total dont 2,0 s de chargement des modèles
(une seule fois) ; sur les 184 979 sessions : 155 alertes règles, 539 IF, 5 826 AE (6 087 au total, 2 597 vrais C2).

## Démo
Montrer un cas **normal** et un cas **suspect**, avec l'explication SHAP de la décision.
