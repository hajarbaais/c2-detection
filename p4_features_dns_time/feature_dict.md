# Dictionnaire des features P4 — DNS & temps

Fichier produit : `p4_features_dns_time/output/features_dns.parquet` — **1 ligne par `session_id`**.
Scripts : `scripts/p4_01_entropy.py` → `p4_05_windows.py` (Hajar), à lancer dans l'ordre.
Tests : `tests/test_p4_features.py` (Wassima) — `python -m pytest tests/test_p4_features.py -v`.

## Lire ce document

| Colonne | Sens |
|---|---|
| **Portée** | `dns` = sessions DNS uniquement (NaN ailleurs) ; `dns+web` = DNS (horodatages des requêtes) **et** HTTPS/HTTP (horodatages des paquets montants de `events_pkts.parquet`) |
| **Intuition C2** | ce qu'on attend d'un canal C2 par rapport au trafic normal |
| **Vide si** | cas où la valeur est NaN / 0 par construction |

> Le préfixe `dns_` est conservé pour toutes les colonnes, **y compris** celles calculées sur les
> sessions web (IAT, périodicité, fenêtres). Le nom ne veut pas dire « DNS seulement » : voir la portée.

## 1. Entropie (P4.1 — `p4_01_entropy.py`)

Sous-domaine = `query` en minuscules, sans le domaine parent (`parent_domain` de P2) et sans les points.
Ex. `x9f2k.a1.google.com` → `x9f2ka1`. Si le nom ne finit pas par le parent (nom DNS-EXF tronqué à
~42 caractères), on garde tout le nom lisible : c'est justement la partie encodée.
Entropie de Shannon : H = −Σ pᵢ·log₂ pᵢ (bits par caractère), pᵢ = fréquence du caractère i.

| Feature | Portée | Définition | Unité / plage | Intuition C2 | Vide si |
|---|---|---|---|---|---|
| `dns_entropy_mean` | dns | moyenne de H(sous-domaine) sur les requêtes de la session | bits, 0 – ~5,2 | élevée : données encodées (base32 ≈ 4,5–5) | — (0 si aucune requête n'a de sous-domaine) |
| `dns_entropy_max` | dns | maximum de H(sous-domaine) | bits | une seule requête encodée suffit | idem |

Valeurs de contrôle (testées) : `""`→0, `aaaa`→0, `www`→0, `ab`/`aabb`→1, `abcd`→2, 32 symboles équiprobables→5.

## 2. Forme des requêtes (P4.2 — `p4_02_shape.py`)

| Feature | Portée | Définition | Unité / plage | Intuition C2 | Vide si |
|---|---|---|---|---|---|
| `dns_qname_len_mean` | dns | longueur moyenne de `query` (caractères) | car. | longue (tunnel) | — |
| `dns_qname_len_max` | dns | longueur max de `query` | car. | idem | — |
| `dns_n_levels_mean` | dns | nb moyen de niveaux (points + 1) | ≥ 1 | beaucoup de niveaux | — |
| `dns_digit_ratio_mean` | dns | moyenne par requête de chiffres / (chiffres + lettres) | 0 – 1 | élevé : encodage | — |
| `dns_pct_txt` | dns | part des requêtes `qtype = 16` (TXT) | 0 – 1 | canal de réponse des tunnels (iodine, dns2tcp) | — |
| `dns_pct_cname` | dns | part `qtype = 5` (CNAME) | 0 – 1 | idem | — |
| `dns_pct_null` | dns | part `qtype = 10` (NULL) | 0 – 1 | iodine | — |
| `dns_unique_ratio` | dns | nb de `query` distinctes / nb de requêtes | 0 – 1 | **1** = exfiltration (tout unique) ; **très bas** = balise répétant le même nom | — |
| `dns_pct_nxdomain` | dns | part des réponses `rcode = 3` (NXDOMAIN) | 0 – 1 | DGA / domaines inexistants | — |

⚠️ `dns_qname_len_*` utilise `len(query)` alors que le README P2 (§6.1) conseille `size` (taille réelle
du paquet), car les noms DNS-EXF sont **tronqués à ~42 caractères** : la longueur sous-estime donc
l'exfiltration. À discuter avec Hajar (ajouter `dns_query_size_mean` basé sur `size`).

## 3. Temps inter-arrivée (P4.3 — `p4_03_iat.py`)

Horodatages : requêtes DNS (`events.parquet`) ; pour HTTPS/HTTP, paquets **montants** (`direction == 'up'`)
de `events_pkts.parquet` (une session DoH = 1 seule connexion, le rythme est au niveau paquet).
IAT = écart entre deux horodatages successifs de la session.

| Feature | Portée | Définition | Unité | Intuition C2 | Vide si |
|---|---|---|---|---|---|
| `dns_iat_mean` | dns+web | moyenne des IAT | s | période de balise | < 2 horodatages |
| `dns_iat_median` | dns+web | médiane des IAT | s | robuste aux pauses | < 2 horodatages |
| `dns_iat_std` | dns+web | écart-type (pandas, ddof = 1) | s | faible : régulier | < 3 horodatages |
| `dns_iat_cv` | dns+web | `std / mean` (0 si mean = 0) | sans unité | **≈ 0** = horloge (balise) ; **≈ 1** = aléatoire (Poisson) ; > 1 = rafales humaines | < 3 horodatages |

## 4. Périodicité (P4.4 — `p4_04_periodicity.py`)

Autocorrélation par FFT : horodatages → comptages par cases (largeur ≈ médiane(IAT)/2, 20 à 4 096 cases)
→ autocorrélation normalisée → pic le plus haut entre le décalage 2 et la moitié de la série.

| Feature | Portée | Définition | Unité / plage | Intuition C2 | Vide si |
|---|---|---|---|---|---|
| `dns_period_dominant` | dns+web | décalage du pic × largeur de case | s | intervalle de balise | NaN si < 10 horodatages, durée nulle ou série constante |
| `dns_periodicity_score` | dns+web | hauteur du pic d'autocorrélation, bornée à [0, 1] | 0 – 1 | **> 0,5** = rythme net, reste au-dessus de l'aléatoire avec 25 % de jitter | **0** si < 10 horodatages (⚠️ 0 veut dire « non mesurable » **ou** « pas de rythme ») |

Valeurs de contrôle (testées) : balise parfaite (5 s, 60 s, 1 h) → score > 0,9 et période exacte ;
jitter ±10 % → score > 0,5, période à ±10 % ; jitter ±25 % > aléatoire ; Poisson → < 0,3.

## 5. Fenêtres longues 1 h / 6 h (P4.5 — `p4_05_windows.py`)

P2 coupe les sessions en fenêtres d'1 h (`window_idx`) reliées par `session_key`. Chaque session reçoit
les mesures de sa clé.

| Feature | Portée | Définition | Unité | Intuition C2 | Vide si |
|---|---|---|---|---|---|
| `dns_persist_hours` | dns+web | nb de fenêtres d'1 h distinctes où la clé est active | h | balise présente des heures/jours | — |
| `dns_span_hours` | dns+web | `max(window_idx) − min(window_idx) + 1` | h | longue durée de vie | — |
| `dns_coverage` | dns+web | `persist_hours / span_hours` | 0 – 1 | ≈ 1 : actif **chaque** heure (balise continue) | — |
| `dns_events_per_hour` | dns+web | événements de la clé / `persist_hours` | évén./h | débit régulier | — |
| `dns_max_events_6h` | dns+web | max des événements sur un bloc de 6 h (`window_idx // 6`) | évén. | pic d'activité sur 6 h | — |

⚠️ **Anomalie connue** (test `test_windows_do_not_mix_pcaps`, en xfail) : le regroupement se fait sur
`session_key` seul. Or la même clé existe dans plusieurs pcaps (16 471 clés sur les données réelles)
et `window_idx` est relatif à chaque pcap → les fenêtres de captures différentes sont mélangées.
Correctif proposé : `groupby(["file", "session_key"])`. Les blocs 6 h sont fixes (`// 6`), pas glissants.

## 6. Couverture et valeurs observées

- `features_dns.parquet` contient 177 781 sessions sur 184 979 : les **7 198 sessions web sans aucun
  paquet de charge utile** n'y figurent pas. ➜ Toujours faire une **jointure gauche** depuis
  `sessions_pcap.parquet` (c'est ce que font `p6_models/scripts/p6_common.py` et le pipeline P7).
- Médianes sur le **train** (DNS : sessions DNS ; autres : toutes sessions présentes), pour repérer
  une feature cassée lors d'une nouvelle exécution :

| Feature | Bénin (médiane) | C2 (médiane) | % NaN (toutes sessions) |
|---|---|---|---|
| `dns_entropy_mean` | 0 | 0.918 | 52 % |
| `dns_entropy_max` | 0 | 1 | 52 % |
| `dns_qname_len_mean` | 13 | 15 | 52 % |
| `dns_qname_len_max` | 14 | 15 | 52 % |
| `dns_n_levels_mean` | 2.33 | 2.25 | 52 % |
| `dns_digit_ratio_mean` | 0 | 0 | 52 % |
| `dns_pct_txt` | 0 | 0 | 52 % |
| `dns_pct_cname` | 0 | 0 | 52 % |
| `dns_pct_null` | 0 | 0 | 52 % |
| `dns_unique_ratio` | 1 | 0.25 | 52 % |
| `dns_pct_nxdomain` | 0 | 0 | 52 % |
| `dns_iat_mean` | 0.173 | 43.1 | 20 % |
| `dns_iat_median` | 0.11 | 0.0692 | 20 % |
| `dns_iat_std` | 0.31 | 56.4 | 43 % |
| `dns_iat_cv` | 1.16 | 1.73 | 43 % |
| `dns_period_dominant` | 0.773 | 233 | 76 % |
| `dns_periodicity_score` | 0 | 0.249 | 4 % |
| `dns_persist_hours` | 1 | 8 | 4 % |
| `dns_span_hours` | 1 | 16 | 4 % |
| `dns_events_per_hour` | 3 | 6 | 4 % |
| `dns_max_events_6h` | 3 | 23 | 4 % |
| `dns_coverage` | 1 | 1 | 4 % |

Le C2 DNS du train est surtout CTU-13 (C2 « au sens large » : requêtes DNS banales de la machine
infectée), d'où des médianes d'entropie / longueur proches du bénin. L'exfiltration DNS-EXF (27 sessions
train) a, elle, une entropie > 4 bits.

## 7. Ne jamais utiliser comme feature

`qname_truncated`, `qname_truncated_share`, `parent_inferred` (artefacts de capture ≈ label),
`src_ip_hash`, `dst_ip_hash`, `session_id`, `session_key`, `file`, `dataset`, `family`, `label`, `split`.
