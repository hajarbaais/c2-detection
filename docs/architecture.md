# Architecture du projet

Voir le README racine pour le pipeline et la correspondance lots ↔ dossiers.

```
PCAP → P2 sessions → P4+P5 features → P6 modèle → P7 démo
                └ P3 contrôle qualité
```

## Flux de données
- P2 produit : sessions_pcap.parquet, events.parquet, events_pkts.parquet, splits/
- P4 produit : features_dns_time.parquet
- P5 produit : features_tls_seq.parquet
- P6 consomme P4+P5 (jointure session_id) → modèle + métriques
- P7 charge le modèle → démo temps réel
