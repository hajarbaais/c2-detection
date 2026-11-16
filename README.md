# C2 Traffic Detection (DNS / HTTPS)

Détection de trafic **Command & Control** caché dans le trafic **DNS** et **HTTPS**,
par Machine Learning sur **métadonnées réseau uniquement** (aucun déchiffrement).
Projet de fin d'études — méthodologie CRISP-DM, équipe de 7 personnes.

## Pipeline

```
PCAP  →  sessions  →  features  →  modèle  →  démo live
 P1       P2          P4 / P5       P6         P7
          └─ P3 : compréhension & contrôle qualité des données
```

| Lot | Dossier | Rôle |
|-----|---------|------|
| **P1** | `p1_acquisition/` | Acquisition & réduction des datasets (CTU-13, DNS-EXF, DoHBrw, Stratosphere) |
| **P2** | `p2_ingestion/` | PCAP → sessions labellisées (tshark), séquences, splits anti-fuite |
| **P3** | `p3_data_understanding/` | Contrôle qualité, labels, EDA, graphes, rapport |
| **P4** | `p4_features_dns_time/` | Features DNS & temps : entropie de Shannon, IAT, périodicité, jitter |
| **P5** | `p5_features_tls_seq/` | Features TLS & séquences : JA3/JA4, certificats, séquences de paquets |
| **P6** | `p6_models/` | Modèles : Autoencoder, LSTM/Transformer, Isolation Forest + évaluation |
| **P7** | `p7_app_demo/` | Démo Streamlit temps réel + explicabilité SHAP + Docker |

## Données (hors dépôt)

Les datasets et les sorties lourdes (`.parquet`, logs) **ne sont pas** dans Git (taille,
limite GitHub de 100 Mo). Ils se partagent par disque / Drive.
Seul l'index **`data/catalog_reduit.csv`** et les petits rapports sont versionnés.

```
data/
├── extracted/          (gitignored) pcaps réduits
├── meta/               c2_endpoints.csv …
└── catalog_reduit.csv  index des fichiers (poussé)
```

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate   |   Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt
```

## Exécution (par étape)

```bash
# P2 — PCAP → sessions (nécessite tshark / Wireshark)
powershell -File p2_ingestion/run_p2.ps1

# P3 — contrôle qualité & rapport
python p3_data_understanding/scripts/p3_suite.py

# P4, P5, P6, P7 : voir le README de chaque dossier
```

## Objectifs (cahier des charges)

- **Recall ≥ 98 %** (attaques détectées)
- **FPR < 0,1 %** (fausses alertes, sur ≥ 100 000 sessions bénignes)
- **Alerte < 5 s**
- **Explicabilité SHAP**

## ⚠️ Limites connues : raccourcis de dataset

Certaines colonnes permettent de reconnaître **la capture d'origine** plutôt que le **comportement C2**.
Un modèle (surtout supervisé) peut les utiliser comme raccourci et afficher de bons scores trompeurs :

| Feature | Constat | Risque |
|---|---|---|
| `session_type == http` | 2 565 C2 contre 262 bénins (91 % de C2) | le modèle apprend « HTTP = C2 » |
| `tls_version == 0` | 66 % des C2 https (2 700 / 4 098) contre 7 % du bénin | reconnaît la capture, pas le comportement |
| `has_ja3` | 0,93 chez le bénin contre 0,48 chez le C2 | idem |

**Sprint 3** : entraîner et évaluer chaque modèle **avec et sans** ces colonnes, puis comparer.
Détails : `docs/note_P5_corrections_imane.md` (section 3) et `p6_models/README.md`.

## ⚠️ Sécurité & conformité

- Les PCAP de malware sont analysés **au niveau réseau uniquement** — **jamais exécutés**.
- Les adresses IP sont **anonymisées** (hachage SHA-256 + sel) — loi 09-08 / RGPD.
- **Ne jamais** pousser `secret_salt.txt` (il permettrait de dé-anonymiser les IP).
