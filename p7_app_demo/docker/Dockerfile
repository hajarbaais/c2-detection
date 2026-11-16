# syntax=docker/dockerfile:1
# P7.6 - Image reproductible : Zeek + plugins JA3/JA4 + tshark + Python (CPU).
# Contexte de build = RACINE du dépôt :  docker build -f p7_app_demo/docker/Dockerfile .
#
# Reproductibilité : tag de Zeek figé (ARG), plugins zkg figés (ARG), paquets Python figés
# (p7_app_demo/docker/requirements-docker.txt). Pour figer aussi l'image de base, ajouter son digest :
#   docker pull zeek/zeek:8.0.9 && docker inspect --format='{{index .RepoDigests 0}}' zeek/zeek:8.0.9
#   puis  FROM zeek/zeek:8.0.9@sha256:<digest>
# Les paquets apt ne sont pas figés individuellement (limite documentée dans docs/docker.md).
ARG ZEEK_VERSION=8.0.9
# Digest de zeek/zeek:8.0.9 (relevé le 16/11/2026). Si ZEEK_VERSION change, mettre à jour ZEEK_DIGEST.
ARG ZEEK_DIGEST=sha256:b705c8932220e5e9e7af3e6519c8b41188aa190066e456f16ab6cfb7d97b760d
FROM zeek/zeek:${ZEEK_VERSION}@${ZEEK_DIGEST}

ARG JA4_VERSION=v0.18.8
ARG JA3_VERSION=master

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/opt/zeek/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 1) Système : Python, tshark (extraction P2), outils de compilation pour zkg
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        tshark \
        g++ cmake make git ca-certificates libpcap-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Plugins Zeek : JA3 (salesforce) et JA4+ (FoxIO). Liste des versions installées conservée dans l'image.
RUN zkg install --force --skiptests zeek/salesforce/ja3 --version ${JA3_VERSION} \
 && zkg install --force --skiptests zeek/foxio/ja4 --version ${JA4_VERSION} \
 && zkg list installed | tee /opt/zeek-packages.txt

# 3) Python : environnement virtuel + dépendances figées
RUN python3 -m venv /opt/venv
COPY p7_app_demo/docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements-docker.txt

# 4) Code du projet (les données, modèles et résultats sont EXCLUS par .dockerignore et montés en volumes)
RUN useradd -m -u 1000 c2 && mkdir -p /app /data /output && chown -R c2:c2 /app /output
WORKDIR /app
COPY --chown=c2:c2 . /app

USER c2
# L'image de base peut définir son propre ENTRYPOINT (zeek) : on le réinitialise pour que CMD soit exécuté tel quel.
ENTRYPOINT []
ENV PYTHONPATH=/app/p6_models/scripts:/app/p6_models/evaluation:/app/p7_app_demo \
    ZEEK_LOG_DIR=/output
CMD ["bash", "p7_app_demo/docker/smoke_test.sh"]
