# P7.6 — Image Docker : Zeek + plugins JA3/JA4 + Python (Oumaima)

- **Tâche** : P7.6, Sprint 2 — « Image reproductible avec versions figées »
- **Fichiers** : `p7_app_demo/docker/{Dockerfile, docker-compose.yml, requirements-docker.txt, smoke_test.sh}`, `.dockerignore` (racine), `.gitattributes`
- **Statut** : image **construite et testée le 16/11/2026** (Windows, Docker Desktop, Docker 29.7.2) — test de fumée : **10 OK, 0 KO**
- **Branche** : `feature/oumaima-sprint2`

## 1. Objectif

Fournir un environnement identique pour toute l'équipe et pour la démonstration, avec :
- **Zeek** pour produire les journaux réseau (conn, dns, ssl, x509) à partir d'un PCAP, sous Linux. P2 avait utilisé tshark parce que Zeek n'était pas installable sous Windows ;
- les plugins **JA3** et **JA4+** (empreintes TLS : colonnes `ja3` et `ja4` de `ssl.log`) ;
- **tshark**, utilisé par les scripts d'extraction de P2 (`p2_01_extract.py`) que l'étape `extract` du pipeline (`p7_app_demo/pipeline.py`, encore à brancher) devra rappeler ;
- **Python** (CPU) avec les dépendances du pipeline : PyTorch, pandas, scikit-learn, pyarrow…

Le tableau de bord Streamlit et l'API (P7.2 à P7.5) seront ajoutés à cette image quand ils existeront.

## 2. Versions figées

| Élément | Comment c'est figé | Version |
|---|---|---|
| Image de base | `zeek/zeek:${ZEEK_VERSION}@${ZEEK_DIGEST}` (ARG du Dockerfile) | Zeek **8.0.9** (LTS), digest `sha256:b705c8932220e5e9e7af3e6519c8b41188aa190066e456f16ab6cfb7d97b760d` |
| Plugin JA4+ | `zkg install zeek/foxio/ja4 --version ${JA4_VERSION}` | **v0.18.8** |
| Plugin JA3 | `zkg install zeek/salesforce/ja3 --version ${JA3_VERSION}` | branche `master`, commit **`502cc6395811c54743b0561419d61900a6df3ff7`** (relevé dans l'image) |
| Python | image de base (Debian) | 3.13.5 |
| Paquets Python | `requirements-docker.txt`, versions exactes | torch 2.15.0 (CPU), pandas 2.2.3, numpy 2.2.0, pyarrow 26.0.1, scikit-learn 1.9.1, joblib 1.6.0 |
| tshark | paquet apt de l'image de base | 4.4.19 (non figé, voir §7) |

La liste des paquets Zeek installés est conservée dans l'image : `/opt/zeek-packages.txt`.
Les versions de l'environnement local de développement sont les mêmes (relevées avec `pip freeze`).

## 3. Construire et tester

Prérequis : Docker Desktop (WSL2 sous Windows) ; un fichier `.gitattributes` contenant `*.sh text eol=lf` (évite que les
fins de ligne Windows cassent `smoke_test.sh`).

```bash
# depuis la racine du dépôt
docker build -f p7_app_demo/docker/Dockerfile -t c2-detection:0.1 .

# test de fumée (Git Bash : MSYS_NO_PATHCONV=1 empêche la conversion des chemins /opt, /app…)
mkdir -p output data/pcaps
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$(pwd)/data/pcaps:/data/pcaps:ro" -v "$(pwd)/output:/output" \
  -v "$(pwd)/p6_models/models:/app/p6_models/models:ro" \
  c2-detection:0.1
```

Avec un PCAP du projet : ajouter `-e PCAP=/data/pcaps/<capture>.pcap`. Sans PCAP, le test utilise l'exemple fourni par le
paquet JA4 (`quic-with-several-tls-frames.pcapng`).

Docker Compose (non testé à ce jour) :
```bash
cd p7_app_demo/docker
PCAP=/data/pcaps/<capture>.pcap docker compose up --build
```

Le test (`smoke_test.sh`) vérifie : versions de Zeek, des plugins, de tshark et de Python ; imports Python ; exécution
**non root** ; rechargement de `autoencoder.pt` (monté en volume) ; Zeek sur un PCAP, avec présence des colonnes `ja3` et
`ja4` dans `ssl.log`. Il retourne un code d'erreur non nul si un contrôle échoue.

## 4. Résultat du test (16/11/2026)

```
== Versions
zeek version 8.0.9
Plugins Zeek (liste figée au build, /opt/zeek-packages.txt) :
zeek/foxio/ja4 (installed: v0.18.8) - Official Zeek package for JA4+ network fingerprinting.
zeek/salesforce/ja3 (installed: master) - JA3 creates 32 character SSL client fingerprints and logs them as a field in ssl.log.
TShark (Wireshark) 4.4.19.
Python 3.13.5
torch      2.15.0+cpu
pandas     2.2.3
numpy      2.2.0
pyarrow    26.0.1
sklearn    1.9.1
joblib     1.6.0

== Contrôles
[OK ] zeek présent
[OK ] tshark présent
[OK ] plugin ja3 installé
[OK ] plugin ja4 installé
[OK ] imports Python (torch, pandas, numpy, pyarrow, sklearn, joblib)
[OK ] exécution non root
[OK ] autoencoder.pt se recharge

== Zeek sur un PCAP
PCAP de test trouvé dans le paquet ja4 : .../ja4/pcap/quic-with-several-tls-frames.pcapng
[OK ] ssl.log produit
[OK ] colonne ja3 dans ssl.log
[OK ] colonne ja4 dans ssl.log
Journaux produits dans /output/smoke : conn.log, packet_filter.log, quic.log, ssl.log
Exemples (ja3, ja4) :
410bc72d10506604b31a74b036fc4bab        q13d0310h3_55b375c5d22e_cd85d2d88918

Résultat : 10 OK, 0 KO
```

## 5. Données et sécurité

- Rien de volumineux ni de sensible n'est dans l'image (`.dockerignore`) : pas de PCAP, de parquet, de modèle `.pt`, de
  `mlflow.db`. Ils sont montés à l'exécution : `data/pcaps` (lecture seule), `p6_models/models` (lecture seule), `output/`.
- Les PCAP ne sont pas conservés dans l'image ; les journaux Zeek vont dans `output/` (hors Git).
- Exécution en **utilisateur non root** (uid 1000). Pour cette raison, `zkg` ne peut pas être exécuté à l'intérieur du
  conteneur : la liste des plugins est lue dans `/opt/zeek-packages.txt`, produite au moment du build.
- Aucun port exposé (le Streamlit n'est pas encore là), aucun secret dans l'image.
- Le `ENTRYPOINT` de l'image Zeek est réinitialisé (`ENTRYPOINT []`) pour exécuter notre commande telle quelle.
- À respecter dans `extract` (P7.1) : aucune commande système construite à partir de données d'entrée.

## 6. Écarts avec le contenu de la tâche

- La tâche mentionne « Zeek + Python + Streamlit » : Streamlit sera ajouté avec P7.5 (le service est prévu, commenté, dans
  `docker-compose.yml`).
- tshark est ajouté parce que le pipeline réutilise les scripts d'extraction de P2.
- `pytest` n'est pas dans l'image pour l'instant (absent de l'environnement local de référence).

## 7. Limites connues

1. Les paquets **apt** (Python système, tshark) ne sont pas figés un par un : la reproductibilité repose sur le tag et le
   digest de l'image de base.
2. **JA3 est installé depuis la branche `master`** : le commit relevé est `502cc63…` ; le verrouiller par `ARG JA3_VERSION`
   n'a pas été testé (`zkg` accepte-t-il un hash de commit ?). Ce paquet est très stable (peu de changements depuis 2020).
3. Les valeurs JA3/JA4 de Zeek peuvent différer de celles de **tshark** utilisées pour construire les features de P2 et P5
   (`ja3_rarity`, `ja4_rarity`). **Non vérifié** : le test n'a utilisé qu'un PCAP d'exemple (QUIC/TLS) du paquet JA4. Il
   faut un petit PCAP du dataset (à demander à P1 ou P2) et comparer `ssl.log` avec les colonnes `ja3` et `ja4` de
   `sessions_pcap.parquet`.
4. **Licence** : JA4 (empreinte TLS cliente) est sous BSD 3-Clause ; les autres méthodes JA4+ (JA4S, JA4H, JA4L, JA4T, …) sont
   sous licence FoxIO, à vérifier avant tout usage hors cadre académique.
5. Le test ne mesure pas les performances : la latence de 5 secondes (P7.7) n'est pas évaluée ici.
6. Image CPU uniquement. Le `docker-compose.yml` n'a pas été exécuté.

## 8. Suite

1. Obtenir un petit PCAP du dataset et comparer les `ja3` / `ja4` de Zeek et de tshark.
2. Brancher `extract` et `sessionize` du pipeline (`p7_app_demo/pipeline.py`) sur Zeek ou tshark dans cette image.
3. Ajouter Streamlit et FastAPI quand `dashboard/app.py` et l'API existent ; mesurer la latence de bout en bout.
4. Tester `docker compose up` et corriger si nécessaire.
