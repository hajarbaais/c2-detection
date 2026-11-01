# P1 — Acquisition & réduction des datasets

Sélectionne, télécharge et **réduit** les datasets à une taille exploitable, tout en gardant
les 2 classes et toutes les familles. Livre à P2 l'index `catalog_reduit.csv`.

## Datasets
CTU-13 · CIC-Bell-DNS-EXF-2021 · CIRA-CIC-DoHBrw-2020 · Stratosphere C2 HTTPS · Stratosphere normal

## Scripts (`scripts/`)
- `p1_extract_logs.py`, `lancer.ps1` — ré-extraction des pcaps bénins complets pour P2
- voir `LISEZMOI.txt`

## Réduction
~93 Go → ~3,5 Go par **troncature des PCAP** + sous-échantillonnage des familles.
Les 5 signaux (Shannon, JA3/JA4, volumes, familles, IAT/jitter) sont préservés.
