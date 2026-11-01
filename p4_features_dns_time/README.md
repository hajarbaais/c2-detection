# P4 — Features DNS & temps

Calcule, **par session**, les features de rythme et de contenu DNS qui séparent le C2 du trafic normal.

## Entrées (depuis P2)
- `p2_ingestion/output/events.parquet` — 1 ligne par événement (requête DNS / connexion) : `ts`, `t_rel`, `size`, `query`, `qtype`, `rcode`, `rtt`
- `p2_ingestion/output/events_pkts.parquet` — 1 ligne par paquet (`ts`, `direction`, `size`) → timing fin des tunnels HTTPS/DoH
- `p2_ingestion/output/sessions_pcap.parquet` — `session_id`, `label`, `family`, `session_type`, `split`

Jointure sur **`session_id`**. Ajuster les scalers **sur `train` uniquement** (voir `splits/`).

## Features à produire
- **Entropie de Shannon** du nom DNS (`query`) — sur la partie lisible (noms DNS-EXF tronqués à ~42 car.)
- **IAT** (inter-arrival times) entre événements : moyenne, écart-type, **coefficient de variation**
- **Périodicité** : autocorrélation / FFT / Lomb-Scargle (détection de beaconing)
- **Jitter** : variation du rythme
- Longueurs / tailles via `size` (pas `len(query)` : noms tronqués)
- Taux de NXDOMAIN (`rcode == 3`), nombre de requêtes, diversité des domaines

## ⚠️ Pièges (d'après P3)
1. **Ne jamais** utiliser `qname_truncated` / `qname_truncated_share` comme feature (artefact = label).
2. Durée + entropie + taille **seules** ne séparent pas les classes → privilégier le **rythme**.
3. Stratosphere = temps **relatif** : les intervalles sont exacts, pas les dates absolues.

## Sortie
`output/features_dns_time.parquet` (1 ligne par `session_id`).
