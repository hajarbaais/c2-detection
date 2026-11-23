# P6.3 LSTM + P6.4 robustesse au jitter (Wassima)

Modèle séquentiel du sprint 3, entraîné sur `features_seq.npz` (P5 : 64 pas × 6 features `log_dt`, `log_size`,
`log_resp_size`, `direction`, `is_dns`, `is_nxdomain`). Il est construit pour être **comparable au Transformer
d'Imane** (`README_transformer.md`) : même sélection de sessions, même seuil, mêmes sorties.

## Fichiers

| Fichier | Rôle |
|---|---|
| `scripts/p6_03_lstm.py` | P6.3 : LSTM (modes `supervised` / `anomaly`) + option P6.4 `--jitter-train` |
| `scripts/p6_04_jitter.py` | P6.4 : jitter réaliste en secondes + évaluation de robustesse de **n'importe quel** `.pt` (LSTM ou Transformer) |
| `scripts/p6_compare_seq.py` | tableau comparatif LSTM / Transformer → `comparison_lstm_transformer.md` |
| `tests/test_p6_lstm_jitter.py` | 11 tests (jitter, masque, padding, bout en bout sur données synthétiques) |

## Ce qui est identique au Transformer (pour que la comparaison soit juste)

- Données : groupes `dns` / `web`, `--subset packets` (web sans sessions « fallback »), `--max-len` (16 pour dns).
- Seuil : bénin de **validation** à 0,1 % de FPR ; le test n'est jamais utilisé pour choisir. Seed 42.
- Entraînement : AdamW (lr 1e-3), lot 256, 30 époques max, arrêt anticipé patience 5 sur la perte de validation.
- Mode `anomaly` : **même goulot z = 16 et même décodeur MLP** que le Transformer. Seule la partie encodeur change
  (LSTM 2 couches, hidden 64, état caché du dernier pas valide, séquences « packées » : le padding n'est jamais lu).
- Sorties : `lstm_<mode>_<subset>_{metrics,by_family,benchmark}.csv` (mêmes colonnes) ; interface `load_lstm()` /
  `score_lstm()` → `lstm_score`, `lstm_alert` pour la fusion d'Oumaima.

## P6.4 : jitter réaliste

P5 stocke `log_dt = log1p(dt)` standardisé. Pour simuler un C2 qui balise « toutes les 60 s ± 50 % », il faut agir
**en secondes** : `dt' = dt × U(1 − j, 1 + j)` (`--jitter-kind down` : `U(1 − j, 1)`, style Cobalt Strike), puis
re-standardiser avec `seq_scaler.pkl`. Le padding, le premier pas (dt = 0) et les 5 autres features ne bougent pas.

- **Entraînement robuste** : `--jitter-train 0.5` → chaque session reçoit un niveau tiré dans U(0 ; 0,5) à chaque lot.
  `anomaly` : entrée bruitée, cible propre. `supervised` : bénin **et** C2 bruités (sinon le modèle apprendrait
  « bruité = C2 »).
- **Évaluation** (automatique à la fin de `p6_03_lstm.py`, ou `p6_04_jitter.py --model <.pt>`) : seules les sessions
  **C2** du test / unseen sont bruitées (c'est l'attaquant qui choisit son jitter) ; bénin et seuil inchangés → la
  FPR ne bouge pas, on mesure la chute du rappel à 0 / 10 / 25 / 50 %. Même protocole pour le Transformer.
- Limite : les séquences web mélangent intervalles entre connexions (le vrai battement) et entre paquets d'une même
  connexion ; on bruite tous les `dt`, donc le test est plus dur que la réalité.

## Lancer (mêmes essais que le Transformer)

```bash
P5=p5_features_tls_seq/output ; P2=<dossier output P2>
# 1. LSTM sans jitter (référence) puis avec jitter à l'entraînement (P6.4)
python -I p6_models/scripts/p6_03_lstm.py --p5 $P5 --p2 $P2 --mode anomaly --groups web --subset packets --out-name lstm_web_anomaly_packets
python -I p6_models/scripts/p6_03_lstm.py --p5 $P5 --p2 $P2 --mode anomaly --groups web --subset packets --jitter-train 0.5 --out-name lstm_web_anomaly_packets_jit50
python -I p6_models/scripts/p6_03_lstm.py --p5 $P5 --p2 $P2 --mode anomaly --groups dns --max-len 16 --out-name lstm_dns_anomaly
python -I p6_models/scripts/p6_03_lstm.py --p5 $P5 --p2 $P2 --mode supervised --groups web --subset all --out-name lstm_web_supervised_all
# 2. même test de jitter sur les .pt d'Imane (Drive)
python -I p6_models/scripts/p6_04_jitter.py --model p6_models/models/tf_web_anomaly_packets.pt --p5 $P5 --p2 $P2
# 3. tableau comparatif
python -I p6_models/scripts/p6_compare_seq.py
```

Windows (PowerShell) : remplacer `$P5` par le chemin complet. Le Transformer dns d'Imane n'a pas de CSV versionné
(`transformer_anomaly_all_metrics.csv` absent) : relancer son essai dns pour remplir la colonne.

## Résultats

Voir `comparison_lstm_transformer.md` (généré, ne pas éditer à la main) et `evaluation/lstm_*_jitter.csv`.
