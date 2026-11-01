# P3 — Compréhension & contrôle qualité des données

Vérifie la fiabilité des données P2 avant l'entraînement (Data Understanding, CRISP-DM).
N'entraîne aucun modèle.

## Entrées
`p2_ingestion/output/sessions_pcap.parquet`, `events.parquet`

## Sorties (`output/`, `graphs/`, `report/`)
- `quality_report.txt`, `suite_report.txt`, `extra_report.txt`
- `labels_a_verifier.csv`, `labels_summary.txt`
- `graphs/` — 8 graphes (durée, events, octets, déséquilibre, taille/entropie DNS)
- `report/Rapport_P3.html` / `.docx`

## 3 risques identifiés (à respecter par P4-P7)
1. **Labels CTU-13** : larges (tout le trafic infecté = C2) → évaluer à part.
2. **Cases vides** (`server_name`, `ja3`) : ne pas utiliser l'absence comme feature.
3. **Accuracy trompeuse** (99,82 % en prédisant "tout bénin") → recall/FPR par famille.
