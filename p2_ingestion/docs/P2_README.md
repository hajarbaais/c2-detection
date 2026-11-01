# P2 — Ingestion PCAP, sessions & découpage

Lot P2 du projet « Détection de trafic C2 (DNS / HTTPS) ». Entrée : la dataset réduite de P1
(`data/`, ~3,6 Go, 372 pcaps). Sortie : sessions labellisées, séquences d'événements, découpage anti-fuite.
Point d'entrée pour P3 (fusion), P4 (features DNS/temps), P5 (features TLS/séquences), P6 (modèles).

## 1. Livrables

| Livrable (fiche P2) | Fichier (`p2/output/`) | Contenu |
|---|---|---|
| Logs dns/ssl/conn/x509 (tâche 1) | `logs/<dataset>/<pcap>/{conn,dns,ssl,x509,pkts}.parquet` | équivalents des journaux Zeek, 1 dossier par pcap (IP en clair : usage interne uniquement) |
| Tables brutes (tâche 2) | idem (Parquet, lisibles par pandas) + `extract_report.csv` | comptes par pcap, 0 erreur sur 372 |
| Table des sessions (tâches 3, 5, 6) | **`sessions_pcap.parquet`** | 184 979 sessions labellisées, IP hachées (v3) |
| Séquences d'événements (tâche 4) | **`events.parquet`** (+ `events_pkts.parquet`) | 749 288 événements / 13 877 478 paquets |
| Fichiers de splits (tâche 7) | **`splits/`** : `train_ids.csv`, `val_ids.csv`, `test_ids.csv`, `unseen_ids.csv`, `split_assignment.csv`, `split_summary.csv` | + colonne `split` dans `sessions_pcap.parquet` |
| Contrôles | `check_report.txt` | 13 vérifications automatiques, toutes OK |

## 2. Reproduire

```powershell
python -m venv p2\.venv ; p2\.venv\Scripts\pip install pandas pyarrow tldextract scikit-learn cryptography
powershell -File p2\run_p2.ps1          # ~15 min sur 4 cœurs
```

| Script | Tâches | Rôle |
|---|---|---|
| `p2_common.py` | — | chemins, ports TLS, résolveurs DoH |
| `p2_01_extract.py` | 1, 2 | tshark -> journaux conn/dns/ssl/x509/pkts (4 processus en parallèle) |
| `p2_labels.py` | 5 | règles de labellisation (mêmes règles que `label_map.map_session` de P1, + ajustement DNS-EXF) |
| `p2_02_sessions.py` | 3, 4, 5, 6 | sessionisation, événements, labels, anonymisation |
| `p2_03_split.py` | 7 | découpage train/val/test/unseen |
| `p2_04_check.py` | — | vérifications -> `check_report.txt` |
| `p2_05_compare_p1.py` | 5 | contrôle croisé avec `label_map.py` de P1 -> `compare_p1_labels.txt` (`python p2_05_compare_p1.py <dossier scripts de P1>`) |

Outils : Windows 10, Wireshark/tshark 4.4.5, Python 3.14, pandas 3.0, pyarrow, tldextract 5.4 (liste des
suffixes embarquée, aucun accès réseau), cryptography (certificats), scikit-learn.

## 3. Écarts avec la fiche et raisons

| Fiche | Fait | Raison |
|---|---|---|
| Zeek + plugin JA3/JA4, zat | **tshark 4.4.5** + pandas | Zeek non disponible sur le poste (Windows, sans WSL/Docker). tshark calcule JA3, JA3S, JA4 en natif. Les journaux produits reprennent les champs utiles de Zeek (`conn`, `dns`, `ssl`, `x509`). Format Parquet au lieu de TSV Zeek, donc `zat` inutile. |
| `label_map.map_session` de P1 | `p2_labels.py` | `scripts/` absent de la première archive ; `label_map.py` reçu ensuite (`scripts.zip`). **Contrôle croisé fait** (`p2_05_compare_p1.py`, sur toutes les sessions avant filtrage) : 185 863 sessions identiques (label et famille) en v3 (111 016 en v1), 218 différences, **toutes dues à l'ajustement DNS-EXF ci-dessous** (203 sessions de bruit Windows C2 chez P1 → retirées ; 15 sessions vers `cicresearch.ca` dans des pcaps bénins, 0 chez P1 → retirées). Aucune différence sur DoHBrw, CTU-13, Stratosphere. |
| Labels DNS-EXF « fichier entier » | seul le domaine d'exfiltration est C2 | Les pcaps d'attaque contiennent aussi le DNS normal de Windows (microsoft.com, windowsupdate.com…). Ces sessions sont **exclues** (None) au lieu d'être comptées C2. Domaine d'exfiltration : `cicresearch.ca` (lu dans les requêtes d'exfiltration non tronquées). |
| Sessions DNS et HTTPS | + sessions **HTTP** (port 80/8080) | Le C2 des botnets CTU-13 (2011) passe surtout en HTTP clair. Colonne `session_type` pour filtrer. |
| — | sessions découpées en **fenêtres de 1 h** | Une balise de plusieurs jours (Stratosphere) donnerait sinon une seule session. Option `--window` (0 = désactivé). `session_key` (haché) relie les fenêtres d'une même clé, ce qui permet à P4 de reconstruire les fenêtres de 6 h. |

## 4. Définitions

**Session** (toujours à l'intérieur d'un seul pcap, puis par fenêtre de 1 h) :

| `session_type` | Clé | Un événement = |
|---|---|---|
| `dns` | (IP client, domaine parent) — domaine parent via tldextract (`a.b.example.co.uk` → `example.co.uk`) | une requête DNS (UDP/TCP 53 ; mDNS/LLMNR/NBNS écartés) |
| `https` | (IP client, IP serveur, port, SNI) — connexion avec handshake TLS **ou** port TLS connu (443, 447, 449, 3889, 4143, 4443, 4743, 8443) | une connexion TCP |
| `http` | (IP client, IP serveur, port) — port 80/8080 sans TLS | une connexion TCP |

**Labels** : `1` = C2, `0` = bénin. Les sessions `None` (non vérifiables) sont **retirées** de toutes les
sorties (comptes dans `sessions_report.csv` : 846 DNS, 180 HTTPS, 76 HTTP).

| Dataset | → 1 (C2) | → 0 (bénin) | → retiré |
|---|---|---|---|
| DoHBrw, pcap malveillant | HTTPS vers un résolveur DoH (1.1.1.1, 9.9.9.11, 8.8.4.4, 176.103.130.130, vérifiés sur les 300 pcaps) port 443 | — | le reste (1 session vers 151.101.2.49, mise à jour système) |
| DoHBrw, pcap bénin | — | tout (`benign_doh` vers un résolveur, sinon `benign_https`) | — |
| DNS-EXF, pcap attaque | DNS vers `cicresearch.ca` | — | autres domaines (203 sessions) |
| DNS-EXF, pcap bénin | — | tout (`benign_dns`) | — |
| CTU-13 | tout (convention des auteurs) | — | — |
| Stratosphere C2 | vers un couple (IP, port) de `meta/c2_endpoints.csv` | — | le reste |
| Stratosphere normal | — | tout | — |

**Anonymisation** : `sha256(sel + IP)[:16]`. Même IP → même empreinte dans tous les fichiers. Le sel est dans
`p2/secret_salt.txt` : **ne pas le publier ni le committer**. `session_key` est aussi haché.

## 5. Schémas

`sessions_pcap.parquet` (1 ligne = 1 session) :

| Colonne | Sens |
|---|---|
| `session_id` | identifiant stable (sha1 de pcap + clé + fenêtre) |
| `label`, `family` | 0/1 et famille (dns2tcp, trickbot, benign_dns…) |
| `session_type` | dns / https / http |
| `dataset`, `file`, `capture_group`, `capture_start` | origine (catalogue P1) |
| `time_reference` | `relative` pour Stratosphere (horodatages commençant en 1970), sinon `absolute` |
| `window_idx`, `session_key` | n° de fenêtre de 1 h, clé de session hachée (commune aux fenêtres) |
| `src_ip_hash`, `dst_ip_hash` | IP hachées — **jamais comme feature** |
| `dst_port`, `server_name` (SNI), `parent_domain` | destination |
| `ts_start`, `ts_end`, `duration`, `n_events` | bornes temporelles et nombre d'événements |
| `qname_truncated_share`, `parent_inferred` | DNS : part de noms tronqués (voir § 6.1) |
| `orig_bytes`, `resp_bytes`, `orig_pkts`, `resp_pkts`, `n_conn_no_payload` | HTTPS/HTTP : volumes (charge utile TCP) |
| `ja3`, `ja4`, `ja3s`, `tls_version`, `cert_self_signed`, `cert_issuer_cn` | TLS (valeur la plus fréquente de la session) |
| `tls_intercepted` | Stratosphere : certificat remplacé par mitmproxy (voir § 6.3) |
| `split`, `unseen_strict` | jeu (train/val/test/unseen) ; voir § 7 |

`events.parquet` (1 ligne = 1 événement, trié par `session_id` puis `seq`) : `seq` (ordre), `ts`, `t_rel`
(secondes depuis le début de session), `event_type` (`dns_query` | `conn`), `size` (DNS : taille de la requête ;
conn : octets montants), `resp_size` (DNS : taille de la réponse ; conn : octets descendants), DNS : `query`,
`qtype`, `rcode` (3 = NXDOMAIN), `answers`, `rtt`, `qname_truncated` ; conn : `duration`, `orig_pkts`, `resp_pkts`.

`events_pkts.parquet` (HTTPS/HTTP, 1 ligne = 1 paquet TCP avec charge utile) : `session_id`, `seq`, `ts`,
`direction` (up/down), `size`. Indispensable pour les tunnels DoH, où une session = **une seule** longue
connexion (1 événement dans `events.parquet`, des milliers de paquets ici).

## 6. Points d'attention pour P3 – P7

1. **DNS-EXF : noms tronqués (snaplen 96 octets).** Les 18 pcaps CIC ne gardent que 96 octets par paquet :
   un nom DNS est coupé après ~42 caractères et Wireshark ne le décode plus. P2 décode la partie lisible à
   partir des octets bruts (`qname_truncated = True`). 99,8 % des requêtes d'exfiltration sont tronquées,
   0 % des requêtes bénignes. Conséquences :
   - **P4** : l'entropie se calcule sur les ~42 premiers caractères (suffisant : c'est la partie encodée) ;
     pour la longueur, utiliser `size` (taille réelle du paquet, intacte) plutôt que `len(query)`.
   - **Ne jamais utiliser `qname_truncated` ni `qname_truncated_share` comme feature** : ce sont des artefacts
     de capture, presque identiques au label sur cette source.
   - Le domaine parent des requêtes tronquées est déduit (`parent_inferred = True`, `cicresearch.ca`).
2. **Forme des sessions DNS** : bénin = beaucoup de domaines différents, peu de requêtes chacun (médiane 2) ;
   exfiltration = un domaine, ~7 000 requêtes par heure. 42 sessions C2 DNS-EXF seulement contre 86 530 bénignes.
3. **Stratosphere C2 : certificats mitmproxy.** `tls_intercepted = True` → ne pas utiliser certificat /
   auto-signé / JA3S (`cert_issuer_cn = mitmproxy`). JA3/JA4 client, volumes et rythme restent authentiques.
4. **CTU-13 = C2 au sens large** : toutes les sessions de la machine infectée sont C2 (convention des auteurs),
   y compris ses requêtes DNS banales (2 260 sessions DNS). P6 peut tester avec et sans.
5. **IP / machine = fuite** : les machines bénignes et infectées diffèrent toujours. Ni `src_ip_hash`, ni
   `dst_ip_hash`, ni `session_key` ne doivent devenir des features.
6. **Horodatages relatifs** Stratosphere (`time_reference = relative`) : les intervalles sont exacts, pas les dates.
7. `n_conn_no_payload` : connexions sans charge utile (SYN sans réponse, scans CTU-13) — à traiter par P3.
8. Les captures DoHBrw ont été réduites par P1 à ~10 min : 13 157 sessions HTTPS bénignes ici
   (contre ~112 000 estimées sur la dataset complète).

## 7. Découpage (tâche 7)

Règles : unité = **machine** pour DoHBrw malveillant (`doh1`…`doh10`, lu dans le nom du fichier), **pcap**
partout ailleurs → deux sessions d'un même pcap ou d'une même machine ne sont jamais dans deux jeux.
Dans chaque strate (intensité d'exfiltration, couple navigateur-résolveur, famille de malware…), les groupes
sont triés par date : les plus anciens en train, puis val, puis test (~70/15/15 % des sessions ; val et test
jamais vides quand la strate a ≥ 3 captures). Détail groupe par groupe : `splits/split_assignment.csv`.

- **Familles non vues** : `dnscat2`, `murlo`, `nsis.ay` → jeu `unseen` (221 sessions, uniquement C2).
  Les 10 machines DoHBrw ont lancé les 3 outils : `unseen_strict = True` (143 sessions) marque les sessions
  non vues dont la machine est aussi en test → ni famille ni machine vues à l'entraînement.
- **cobaltstrike** → test (5 sessions de 1 h = 250 connexions de balisage).
- **firefox-adguard, firefox-cloudflare** (DoHBrw bénin, une seule capture de 5 Go chacune) → test, pour le
  volume FPR (v3, voir plus bas). `firefox-quad9` et `firefox-google/small` restent en train.
- Pour mesurer le FPR, utiliser les sessions bénignes de `test` (le jeu `unseen` ne contient pas de bénin).

| Jeu | Bénin | C2 |
|---|---|---|
| train | 56 154 | 3 343 |
| val | 11 957 | 1 244 |
| test | 107 903 | 4 157 |
| unseen | 0 | 221 |

Par source :

| Dataset | train 0 / 1 | val 0 / 1 | test 0 / 1 | unseen 1 |
|---|---|---|---|---|
| DNS-EXF | 46 883 / 27 | 9 602 / 8 | 30 045 / 7 | 0 |
| DoHBrw | 8 097 / 143 | 2 095 / 18 | 76 734 / 39 | 100 |
| CTU-13 | 0 / 2 451 | 0 / 297 | 0 / 2 037 | 121 |
| Stratosphere C2 | 0 / 722 | 0 / 921 | 0 / 2 074 | 0 |
| Stratosphere normal | 1 174 / 0 | 260 / 0 | 1 124 / 0 | 0 |

Les proportions s'écartent de 70/15/15 parce qu'on ne coupe jamais un pcap : certains pcaps DNS-EXF bénins
pèsent ~13 000 sessions chacun.

**v2 — PCAP bénins complets (24/10/2026).** Les 6 PCAP bénins du jeu test étaient tronqués (~62 Mo) dans la
dataset réduite. P1 a extrait les journaux des versions complètes (8,6 Go, sha256 vérifiés contre le catalogue)
avec `pour_P1/p1_extract_logs.py` (même code que `p2_01_extract.py` ; le PCAP firefox-google/large de 5 Go est
découpé en morceaux de 1,5 M paquets avec editcap puis recollé, ce qui coupe en deux les rares connexions à
cheval sur une frontière). Rapport : `output/p1_extract_report.csv`. Effet : +35 562 sessions bénignes, toutes en
test sauf `chrome-quad9/dump_00007` (420 sessions) qui passe de val à train, car le découpage pondère par le
nombre de sessions. Aucun autre groupe ne change de jeu ; les 13 contrôles restent OK et les labels des nouvelles
sessions concordent à 100 % avec `label_map.py` de P1. Version précédente conservée dans `output_v1/`.

**v3 — volume FPR atteint (25/10/2026).** P1 a extrait de la même façon les 2 PCAP bénins firefox-adguard et
firefox-cloudflare complets (5,1 Go chacun, sha256 vérifiés ; `output/p1_extract_report_lot2.csv`). Chacun est la
seule capture de sa strate, donc la règle 70/15/15 l'envoyait en train : ils sont forcés en test
(`FORCE_TEST_FILES` dans `p2_03_split.py`), comme cobaltstrike. Seuls ces 2 fichiers changent de jeu par rapport à la
v2 (train perd leurs 722 sessions tronquées). Versions précédentes : `output_v1/`, `output_v2/`.

**✅ FPR < 0,1 %** : **107 903 sessions bénignes en test** (≥ 100 000 demandées). Règle de trois : 0 faux positif
→ FPR < 3 / 107 903 ≈ 0,0028 % (IC 95 %).
