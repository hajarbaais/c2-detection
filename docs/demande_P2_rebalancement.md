# Demande à P2 : rééquilibrer le découpage train / val / test

- **De** : Oumaima (autoencodeur, P5.6) — **Pour** : P2 (ingestion, sessions, split), copie Wassima (P6), Imane (P5), P3
- **Date** : 13/11/2026
- **Pourquoi maintenant** : le LSTM (sprint 3) et la fusion vont être entraînés sur ce découpage. Le corriger avant évite de tout refaire.

## 1. Constat (mesuré sur `sessions_pcap.parquet`, identique dans `sessions_unified.parquet`)

**1.1 Le bénin web est presque entièrement dans le test**

| | train | val | test |
|---|---|---|---|
| Sessions bénignes web | 8 836 (10 %) | 2 268 (2,6 %) | 77 591 (87,5 %) |
| Pcaps bénins web | 22 | 6 | 8 |

Les 8 pcaps du test contiennent en moyenne ~9 700 sessions, les 22 du train ~400 : découper par pcap entier,
avec des pcaps de tailles très inégales, a produit ce déséquilibre. Le DNS est correct (47 318 / 9 689 / 30 312).

Dans le détail, `benign_doh` n'a que **21 sessions en train** et 16 en val, contre 999 en test : le modèle n'a presque jamais vu de DoH bénin.

**1.2 Le bénin web vient presque d'une seule source** : 98 % de CIRA-CIC-DoHBrw (86 926 sessions), 2 % de
Stratosphere-normal (1 769). Un modèle peut apprendre « DoHBrw = normal ».

**1.3 Conséquences observées (autoencodeur `ae_v2`)** : FPR web de 0,09 % en validation mais **4,13 % en test**
(`benign_doh` : 61,6 % de fausses alertes). La validation web ne permet pas de calibrer 0,1 % : 2 268 × 0,1 % = 2,3
fausses alertes autorisées. Wassima l'a observé aussi (Isolation Forest : FPR test 0,77 %, validation 0,09 %).

**1.4 Les familles C2 web sont réparties de façon inégale**

| Famille | train | val | test | unseen |
|---|---|---|---|---|
| rbot | **2 058** | 4 | 12 | – |
| trickbot | 236 | 921 | **1 469** | – |
| dridex | 486 | **0** | 600 | – |
| neris | 88 | 159 | 184 | – |
| dns2tcp / iodine | 73 / 70 | 8 / 10 | 19 / 20 | – |
| virut | 36 | 0 | 37 | – |
| dnscat2 / murlo / nsis.ay | – | – | – | 100 / 13 / 40 |

Le rappel mesuré dépend donc de la composition de chaque ensemble : 13 % sur le train, 62,6 % en val, 55,8 % en test,
pour le même modèle. Un modèle supervisé ou le LSTM entraîné sur ce train verrait surtout du rbot (67 % des C2 web du train).

**1.5 Labels DNS « par convention »** : 1 804 des 1 811 C2 DNS du test sont des requêtes DNS banales de la machine
infectée de CTU-13 (`ctu13_broad_label`). Les sessions d'exfiltration DNS ne sont que 49 en tout (27 train, 8 val, 7 test).

**1.6 `unseen`** : 221 sessions, dont 143 `unseen_strict` (68 DNS, 75 web). Les 68 DNS sont toutes des sessions DNS banales
de CTU-13, donc leur rappel n'est pas interprétable.

**1.7 Test de l'hypothèse (expérience `ae_resplit`, 13/11/2026)** : en redécoupant nous-mêmes le bénin web par pcap et par
famille (60 / 15 / 21 %, écart de 2 %), sans toucher au modèle, le FPR web de test passe de **4,13 % à 0,56 %**, les
fausses alertes sur `benign_doh` de 61,6 % à 0 %, et la validation (13 906 sessions) devient fiable. Ce résultat est une
borne optimiste (mêmes machines, sessions voisines dans le temps), mais il confirme que le découpage actuel est la cause
principale. Détail : `docs/p5_6b_autoencoder_train.md`, section 6 bis.

## 2. Demandes, par priorité

### Priorité 1 : rééquilibrer le bénin web (bloquant)
- **Objectif** : train ≥ 55 % du bénin web, val **≥ 10 000 sessions** (0,1 % = 10 fausses alertes, seuil fiable), test ~20–25 %.
- **Méthode proposée** : pour les gros pcaps aujourd'hui entièrement dans le test, faire un **découpage temporel
  à l'intérieur du pcap** (début → train, milieu → val, fin → test) avec un **intervalle d'écart** (par exemple 1 h, comme les
  fenêtres de sessions) pour qu'aucune session ou fenêtre ne soit à cheval. Garder un pcap entier dans un seul
  ensemble reste la règle pour les petits pcaps.
- **Contrainte à documenter** : les mêmes machines seront dans train, val et test (c'est déjà le cas, voir P3). Le dire
  explicitement dans le README : la généralisation à de nouvelles machines ne sera pas évaluée.
- **Estimation du FPR** : 20 000 sessions bénignes web en test suffisent pour estimer un FPR de 0,1 % (en dessous de 3 000
  sessions à 0 faux positif on ne peut pas conclure). Le seuil de « 100 000 bénins » donné dans le guide s'applique à
  l'ensemble, pas à chaque groupe.

### Priorité 1 : rééquilibrer les familles C2 web
- Pour chaque famille de plus de 500 sessions (rbot, trickbot, dridex, neris), viser **au moins 15 % dans la validation et 15 %
  dans le test**, en répartissant les pcaps (ou les fenêtres temporelles) entre les ensembles.
- Pour les petites familles (dns2tcp, iodine, virut, cobaltstrike), au moins une session dans chaque ensemble quand c'est possible.
- Conserver 3 familles entièrement hors train, listées dans le README, comme ensemble `unseen`.

### Priorité 1 : rendre le label DNS exploitable
- Intégrer `ctu13_broad_label` (P3) dans la table officielle de P2, et fournir une colonne `label_strict` : C2 = 1
  uniquement pour les sessions réellement C2 (exfiltration et tunnels), 0 ailleurs, ou exclu.
- Fournir un `split_summary.csv` qui donne aussi les effectifs **avec et sans** ces sessions.

### Priorité 2 : diversifier le bénin web
- Vérifier si les gros pcaps bénins extraits par P1 (`firefox-adguard`, `firefox-cloudflare`, 8,6 Go et 10 Go) sont bien
  dans `sessions_pcap.parquet`, et dans quel ensemble.
- Ajouter si possible une autre source de bénin HTTPS/TLS que DoHBrw (captures de navigation, jeux CIC ou Stratosphere
  normaux), pour que le bénin ne soit pas lié à un seul dataset (risque de raccourci signalé par P3 : 99,82 % d'accuracy en
  devinant la source).

### Priorité 2 : plus de DNS tunneling et un `unseen` plus solide
- 49 sessions d'exfiltration DNS : un rappel de 100 % sur 7 sessions a une borne basse de 65 %. Chercher d'autres captures
  de dnscat2, iodine, dns2tcp (jeux publics, ou génération en laboratoire autorisé et isolé), avec labels documentés.
- Viser `unseen_strict` ≥ 300 sessions, ≥ 3 familles, avec des labels non conventionnels.

## 3. Ce qui ne doit pas changer
`session_id`, `label`, `family`, les champs de session et les événements : seules la colonne `split` (et `unseen_strict`)
changent. Cela évite de refaire l'extraction : il suffit de relancer `p2_03_split.py` et `p2_04_check.py`, sauf si de
nouvelles captures sont ajoutées.

## 4. Livraison attendue
1. `sessions_pcap.parquet` avec un nouveau `split` et une colonne `split_version` (par exemple `v4`).
2. `splits/*.csv` et `split_summary.csv` mis à jour (effectifs par groupe dns/web, par famille, par ensemble).
3. `check_report.txt` : aucun session d'un même pcap ou d'une même fenêtre dans deux ensembles (hors découpage
   temporel avec écart), 0 fuite.
4. Un paragraphe dans le README P2 : méthode du découpage, machines partagées, limites.

**Critères d'acceptation** : bénin web train ≥ 55 %, val ≥ 10 000, test 20–25 % ; chaque famille de plus de 500 sessions
présente dans les trois ensembles ; `unseen_strict` ≥ 300 ; versions documentées.

## 5. Ce que chaque équipe doit relancer après la livraison

| Qui | À relancer |
|---|---|
| P3 | rapport qualité sur la nouvelle table |
| P4 (Hajar) | rien : les features ne dépendent pas du split (vérifier le regroupement par `(file, session_key)`) |
| P5 (Imane) | `p5_features.py` : le scaler, `ja3_rarity`, `ja4_rarity` et la réputation dépendent du train |
| P6 (Wassima) | règles calibrées, Isolation Forest, HistGradientBoosting |
| Oumaima | `p5_6b_autoencoder_train.py` (nouveau run MLflow) |
| LSTM | à entraîner **après** cette livraison |

## 6. Pourquoi cela concerne aussi le LSTM
Le web est le seul groupe avec de vraies séquences (HTTPS : ~32 paquets en moyenne ; DNS : 2,2 pas). Si le LSTM est
supervisé, il verra un train dominé par rbot (67 % des C2 web) et un bénin à 90 % issu d'un autre profil que le test.
Il aura donc les mêmes limites que l'autoencodeur, quelle que soit sa qualité.
