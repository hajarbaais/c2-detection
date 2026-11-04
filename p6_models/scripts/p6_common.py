"""P6 - fonctions communes aux scripts de modèles (chargement, matrice de features, MLflow).

Utilisé par p6_01_rule_baselines.py, p6_02_isolation_forest.py et p7_app_demo/pipeline.py.

Données lues :
    <P2>/sessions_pcap.parquet                       labels, split, session_type, volumes bruts
    p4_features_dns_time/output/features_dns.parquet  features de Hajar (P4)
    (plus tard) p5_features_tls_seq/output/...        features d'Imane (P5) -> ajouter dans load_features()
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_P2 = Path.home() / "Desktop" / "P2_livrables_v3" / "P2_livrables_v3" / "output"
P4_FEATURES = REPO / "p4_features_dns_time" / "output" / "features_dns.parquet"
MODELS_DIR = REPO / "p6_models" / "models"
EVAL_DIR = REPO / "p6_models" / "evaluation"

TARGET_FPR = 0.001      # 0,1 % de fausses alertes (cahier des charges)
TARGET_RECALL = 0.98
RANDOM_STATE = 42

# Colonnes de sessions_pcap.parquet utilisables comme features (volumes bruts, en attendant P5)
SESSION_NUMERIC = ["duration", "n_events", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
                   "n_conn_no_payload"]

# Ne JAMAIS utiliser comme feature : identifiants, artefacts de capture, fuite de label (README P2 §6)
FORBIDDEN = {"session_id", "session_key", "src_ip_hash", "dst_ip_hash", "label", "family", "dataset",
             "file", "capture_group", "capture_start", "split", "unseen_strict", "time_reference",
             "window_idx", "ts_start", "ts_end", "qname_truncated", "qname_truncated_share",
             "parent_inferred"}

# Groupes de modèles : un modèle par famille de trafic (DNS et web n'ont pas les mêmes features)
GROUPS = {"dns": ["dns"], "web": ["https", "http"]}


def p2_dir(arg: str | None) -> Path:
    """Dossier des sorties P2 : argument --p2, sinon variable C2_P2_DIR, sinon le chemin par défaut."""
    return Path(arg or os.environ.get("C2_P2_DIR") or DEFAULT_P2)


def load_dataset(p2: Path, features_path: Path = P4_FEATURES) -> pd.DataFrame:
    """1 ligne par session : métadonnées P2 + features P4 (jointure GAUCHE sur session_id).

    Jointure gauche : ~7 200 sessions web sans paquet de charge utile n'ont aucune ligne dans
    features_dns.parquet ; elles restent dans le jeu (features à NaN) au lieu de disparaître.
    """
    sessions = pd.read_parquet(p2 / "sessions_pcap.parquet")
    if not features_path.exists():
        sys.exit(f"[!] {features_path} introuvable : lancer d'abord les scripts P4 (p4_01 ... p4_05).")
    feats = pd.read_parquet(features_path)
    df = sessions.merge(feats, on="session_id", how="left", validate="one_to_one")
    df["group"] = df["session_type"].map({t: g for g, ts in GROUPS.items() for t in ts})
    return df


def feature_columns(df: pd.DataFrame, group: str) -> list[str]:
    """Colonnes numériques autorisées pour un groupe (dns -> features dns_* ; web -> volumes + rythme)."""
    p4 = [c for c in df.columns if c.startswith("dns_") and c not in FORBIDDEN]
    if group == "dns":
        cols = p4 + ["duration", "n_events"]
    else:
        # pour le web, les features de contenu DNS sont toujours vides : on garde rythme + fenêtres
        rythm = [c for c in p4 if any(k in c for k in ("iat", "period", "persist", "span",
                                                        "events_per_hour", "max_events", "coverage"))]
        cols = rythm + SESSION_NUMERIC
    cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    assert not (set(cols) & FORBIDDEN), "colonne interdite dans les features"
    return cols


# Colonnes à queue lourde -> log1p avant Isolation Forest
HEAVY = ("bytes", "pkts", "n_events", "duration", "iat", "period_dominant", "span", "events_per_hour",
         "max_events", "len", "no_payload")


def to_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Matrice numérique prête pour sklearn : log1p sur les colonnes à queue lourde, NaN -> -1."""
    X = df[cols].astype("float64").copy()
    for c in cols:
        if any(h in c for h in HEAVY):
            X[c] = np.log1p(X[c].clip(lower=0))
    return X.fillna(-1.0).to_numpy()


# --------------------------------------------------------------------------- #
# MLflow
# --------------------------------------------------------------------------- #
def setup_mlflow(experiment: str):
    """Configure MLflow et renvoie le module (ou None si mlflow n'est pas installé).

    Backend par défaut : sqlite:///<repo>/mlflow.db (+ artefacts dans <repo>/mlruns/), commun à
    tous les scripts. Surcharger avec la variable MLFLOW_TRACKING_URI (ex. serveur partagé).
    Voir docs/mlflow.md pour les conventions de l'équipe.
    """
    try:
        import mlflow
    except ImportError:
        warnings.warn("mlflow non installé : pip install mlflow  (les runs ne seront pas enregistrés)")
        return None
    uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(REPO / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(REPO / "mlruns").as_uri())
    mlflow.set_experiment(experiment)
    return mlflow


def git_commit() -> str:
    """Hash du commit courant (tag MLflow : relier chaque run au code exact)."""
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"
