"""P6 - Features TLS de P5 (Imane) pour les modèles P6 et le pipeline P7.

Problème corrigé ici : dans features_tls.parquet (P5 v1), la rareté JA3/JA4 et la réputation de
domaine sont calculées sur TOUT le train, C2 compris. Les empreintes des C2 du train paraissent donc
« courantes ». Oumaima a déjà recalculé ces 3 colonnes sur le bénin du train web (P5.6b) ; ce module
fait EXACTEMENT le même calcul, mais :
  - en deux temps (fit sur le bénin du train / transform sur n'importe quelle session),
  - avec une référence sauvegardée (models/tls_reference.joblib) que le pipeline recharge une fois,
    ce qui permet de scorer une nouvelle capture sans relire le train.

Formules (identiques à recompute_p5_on_benign de p5_6b_autoencoder_train.py) :
  rareté     = -log(fréquence dans le bénin du train web) ; valeur jamais vue -> -log(1/(n_ref+1)) ;
               pas de JA3/JA4 -> 0
  réputation = log1p(nb de machines distinctes du bénin du train web qui contactent le domaine)
               (clé = server_name, sinon dst_ip_hash)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 14 features tabulaires de P5, préfixées p5_ à la jointure (sessions_pcap contient déjà
# tls_version et cert_self_signed bruts : on ne les écrase pas)
P5_COLS = ["log_orig_bytes", "log_resp_bytes", "up_down_ratio", "avg_pkt_size", "log_duration",
           "log_n_conn", "log_n_conn_no_payload", "conn_rate", "has_ja3", "ja3_rarity", "ja4_rarity",
           "tls_version", "cert_self_signed", "log_domain_reputation"]
P5_FEATS = ["p5_" + c for c in P5_COLS]
RECOMPUTED = ["p5_ja3_rarity", "p5_ja4_rarity", "p5_log_domain_reputation"]
REF_FILE = "tls_reference.joblib"


def reference_mask(df: pd.DataFrame) -> pd.Series:
    """Sessions qui servent de référence : bénin du train, groupe web."""
    return (df["split"] == "train") & (df["label"] == 0) & (df["group"] == "web")


def _domain_key(df: pd.DataFrame) -> pd.Series:
    return df["server_name"].fillna(df["dst_ip_hash"])


def fit_tls_reference(df: pd.DataFrame) -> dict:
    """Apprend les tables de fréquence sur le bénin du train web uniquement."""
    ref = reference_mask(df)
    key = _domain_key(df)
    return {
        "n_ref": int(ref.sum()),
        "ja3_freq": df.loc[ref, "ja3"].value_counts(normalize=True).to_dict(),
        "ja4_freq": df.loc[ref, "ja4"].value_counts(normalize=True).to_dict(),
        "domain_hosts": df.loc[ref].groupby(key[ref])["src_ip_hash"].nunique().to_dict(),
    }


def _rarity(col: pd.Series, freq: dict, n_ref: int) -> pd.Series:
    max_rar = -np.log(1.0 / (n_ref + 1))
    rar = (-np.log(col.map(freq).astype("float64"))).fillna(max_rar)
    rar[col.isna()] = 0.0
    return rar


def apply_tls_reference(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Remplace les 3 colonnes dépendantes du train par leur version « bénin du train »."""
    df = df.copy()
    df["p5_ja3_rarity"] = _rarity(df["ja3"], ref["ja3_freq"], ref["n_ref"])
    df["p5_ja4_rarity"] = _rarity(df["ja4"], ref["ja4_freq"], ref["n_ref"])
    df["p5_log_domain_reputation"] = np.log1p(_domain_key(df).map(ref["domain_hosts"]).astype("float64")
                                              .fillna(0))
    # ces colonnes n'ont de sens que pour le web (comme dans features_tls.parquet)
    df.loc[df["group"] != "web", RECOMPUTED] = np.nan
    return df


def merge_p5(df: pd.DataFrame, p5_dir: Path | str) -> pd.DataFrame:
    """Jointure GAUCHE des 14 features de features_tls.parquet (préfixe p5_)."""
    t = pd.read_parquet(Path(p5_dir) / "features_tls.parquet", columns=["session_id"] + P5_COLS)
    t = t.rename(columns={c: "p5_" + c for c in P5_COLS})
    return df.drop(columns=[c for c in P5_FEATS if c in df.columns]).merge(
        t, on="session_id", how="left", validate="one_to_one")


def save_reference(ref: dict, models_dir: Path) -> Path:
    import joblib
    path = Path(models_dir) / REF_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ref, path)
    return path


def load_reference(models_dir: Path) -> dict | None:
    import joblib
    path = Path(models_dir) / REF_FILE
    return joblib.load(path) if path.exists() else None
