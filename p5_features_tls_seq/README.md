# P5 — Features TLS & séquences

Features TLS et séquentielles par session.

## Entrées (depuis P2)
`sessions_pcap.parquet` (ja3, ja4, ja3s, tls_version, cert_self_signed, cert_issuer_cn,
tls_intercepted) + `events_pkts.parquet` (séquences de paquets).

## Features
- Empreintes **JA3 / JA4** (clientes), JA3S (serveur)
- Version TLS, certificat auto-signé, SAN, validité (si logs x509 fournis)
- Séquences de tailles/direction de paquets (pour LSTM/Transformer de P6)

## ⚠️ Pièges (P3)
- Ne pas utiliser `server_name`/`ja3` **vides** comme critère (fuite).
- Stratosphere `tls_intercepted = True` : ignorer certificat / auto-signé / JA3S
  (remplacés par mitmproxy). JA3/JA4 client, volumes et rythme restent valides.

## Sortie
`output/features_tls_seq.parquet` (1 ligne par `session_id`).
