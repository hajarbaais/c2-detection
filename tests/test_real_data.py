"""Tests sur les VRAIES données du projet (sorties P2 du Sprint 1 + features P4), pas sur des données inventées.

Les autres fichiers de tests/ utilisent une mini-sortie P2 synthétique (rapides, réponses connues).
Ici on vérifie que le code et les données réelles respectent les règles du projet : volumes, anti-fuite,
familles non vues, plages des features, et que les premiers modèles font mieux que le hasard.

Données attendues (hors dépôt, trop lourdes pour Git) :
    <P2>/sessions_pcap.parquet                          dossier donné par C2_P2_DIR (sinon chemin par défaut de p6_common)
    p4_features_dns_time/output/features_dns.parquet     ou chemin donné par C2_P4_FEATURES

Lancer (Windows, depuis la racine du dépôt) :
    set C2_P2_DIR=D:\\chemin\\vers\\P2_livrables_v3\\output
    python -m pytest tests/test_real_data.py -v
Sans données, les tests sont IGNORÉS (skipped) avec le message qui dit quoi fournir ; ils n'échouent pas.
"""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import p6_common
from p6_common import FORBIDDEN, TARGET_FPR, p2_dir

REPO = Path(__file__).resolve().parents[1]
P2 = p2_dir(None)
SESSIONS = P2 / "sessions_pcap.parquet"
P4_FEATS = Path(os.environ.get("C2_P4_FEATURES") or p6_common.P4_FEATURES)
SPLIT_REPORT = REPO / "p2_ingestion" / "output" / "splits" / "split_summary.csv"

needs_p2 = pytest.mark.skipif(not SESSIONS.exists(),
                              reason=f"données réelles absentes : {SESSIONS} (définir C2_P2_DIR)")
needs_p4 = pytest.mark.skipif(not (SESSIONS.exists() and P4_FEATS.exists()),
                              reason=f"features réelles absentes : {P4_FEATS} (lancer les scripts P4 ou définir C2_P4_FEATURES)")


@pytest.fixture(scope="module")
def sessions():
    return pd.read_parquet(SESSIONS)


@pytest.fixture(scope="module")
def dataset():
    """Sessions P2 + features P4 réelles (même chargement que les modèles)."""
    return p6_common.load_dataset(P2, P4_FEATS)


# --------------------------------------------------------------------------- #
# 1. Données réelles du Sprint 1 (sortie P2)
# --------------------------------------------------------------------------- #
@needs_p2
def test_session_id_unique(sessions):
    assert sessions["session_id"].is_unique


@needs_p2
def test_no_clear_ip_column(sessions):
    assert not {"src_ip", "dst_ip"} & set(sessions.columns), "IP en clair : anonymisation cassée"


@needs_p2
def test_volumes_match_p2_report(sessions):
    """Les données reçues sont bien celles du rapport P2 versionné (bénin / C2 par jeu et par dataset)."""
    ref = pd.read_csv(SPLIT_REPORT).set_index(["split", "dataset"]).sort_index()
    got = (sessions.groupby(["split", "dataset"])["label"]
           .agg(benign=lambda s: int((s == 0).sum()), c2=lambda s: int((s == 1).sum())).sort_index())
    pd.testing.assert_frame_equal(got, ref[["benign", "c2"]], check_dtype=False)


@needs_p2
def test_enough_benign_to_measure_fpr(sessions):
    """Cahier des charges : FPR mesurée sur au moins 100 000 sessions bénignes (jeu test)."""
    assert ((sessions["split"] == "test") & (sessions["label"] == 0)).sum() >= 100_000


@needs_p2
def test_no_pcap_in_two_splits(sessions):
    """Anti-fuite : un même pcap n'est jamais coupé entre train, val et test."""
    n_splits = sessions.groupby("file")["split"].nunique()
    assert (n_splits == 1).all(), f"pcaps dans plusieurs jeux : {list(n_splits[n_splits > 1].index[:5])}"


@needs_p2
def test_unseen_families_never_in_training(sessions):
    """Généralisation : les familles du jeu « unseen » n'apparaissent ni en train ni en val."""
    unseen = set(sessions.loc[sessions["split"] == "unseen", "family"])
    seen = set(sessions.loc[sessions["split"].isin(["train", "val"]), "family"])
    assert unseen and not (unseen & seen), f"familles vues à l'entraînement : {unseen & seen}"


@needs_p2
def test_unseen_contains_only_c2(sessions):
    assert (sessions.loc[sessions["split"] == "unseen", "label"] == 1).all()


# --------------------------------------------------------------------------- #
# 2. Features P4 calculées sur les vraies données
# --------------------------------------------------------------------------- #
@needs_p4
def test_every_real_dns_session_has_features(sessions):
    feats = pd.read_parquet(P4_FEATS, columns=["session_id"])
    dns_ids = set(sessions.loc[sessions["session_type"] == "dns", "session_id"])
    missing = dns_ids - set(feats["session_id"])
    assert not missing, f"{len(missing)} sessions DNS sans features"


@needs_p4
def test_no_forbidden_column_in_real_features():
    cols = set(pd.read_parquet(P4_FEATS).columns) - {"session_id"}
    assert not (cols & FORBIDDEN), f"colonnes interdites : {cols & FORBIDDEN}"


@needs_p4
def test_real_feature_ranges(dataset):
    """Plages documentées dans feature_dict.md : entropie en bits (alphabet DNS), score de périodicité dans [0, 1]."""
    ent = dataset["dns_entropy_max"].dropna()
    assert ent.between(0, math.log2(64)).all()
    per = dataset["dns_periodicity_score"].dropna()
    assert per.between(0, 1).all()


@needs_p4
def test_real_exfiltration_has_higher_entropy_than_benign(dataset):
    """Signal réel : l'exfiltration DNS (DNS-EXF) a des sous-domaines plus aléatoires que le DNS bénin."""
    dns = dataset[dataset["session_type"] == "dns"]
    exfil = dns.loc[(dns["label"] == 1) & dns["family"].str.startswith("dns_exfil"), "dns_entropy_max"].dropna()
    benign = dns.loc[dns["label"] == 0, "dns_entropy_max"].dropna()
    assert len(exfil) > 0 and len(benign) > 0
    assert exfil.median() > benign.median()


# --------------------------------------------------------------------------- #
# 3. Premiers modèles sur les vraies données
# --------------------------------------------------------------------------- #
@needs_p4
def test_calibrated_rules_respect_fpr_on_real_train(dataset):
    from p6_01_rule_baselines import predict, rule_scores, thresholds
    sc = rule_scores(dataset)
    pred = predict(sc, thresholds(sc, dataset, "calibrated"))
    benign_train = (dataset["split"] == "train") & (dataset["label"] == 0)
    assert pred.loc[benign_train, "R_any"].mean() <= TARGET_FPR


@needs_p4
def test_iforest_better_than_random_on_real_val(dataset):
    """Isolation Forest entraîné sur le bénin du train : il classe mieux que le hasard sur la validation DNS."""
    pytest.importorskip("sklearn.ensemble")
    from metrics import roc_auc_score
    from p6_02_isolation_forest import score, train_iforest
    b = train_iforest(dataset, "dns")
    val = dataset[(dataset["split"] == "val") & (dataset["group"] == "dns")]
    assert roc_auc_score(val["label"], score(b, val)) > 0.7
