"""Tests P6 (Wassima) : métriques, baselines à règles, Isolation Forest, + pipeline P7 (squelette)."""
import math

import numpy as np
import pandas as pd
import pytest

from metrics import binary_metrics, threshold_at_fpr, wilson_ci, metrics_by
import p6_common
from p6_01_rule_baselines import RULES, predict, rule_scores, thresholds


# --------------------------------------------------------------------------- métriques
class TestMetrics:
    def test_threshold_at_fpr_respects_budget(self):
        benign = np.arange(10_000, dtype=float)
        thr = threshold_at_fpr(benign, 0.001)
        assert (benign > thr).mean() <= 0.001
        assert (benign > thr).sum() == 10                   # budget utilisé entièrement

    def test_threshold_ignores_nan_and_empty(self):
        assert threshold_at_fpr(np.array([np.nan, 1.0, 2.0]), 0.0) == 2.0
        assert threshold_at_fpr(np.array([]), 0.001) == math.inf

    def test_binary_metrics_counts(self):
        y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        p = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
        m = binary_metrics(y, p)
        assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
        assert m["recall"] == 0.75 and m["fpr"] == pytest.approx(1 / 6)
        assert m["precision"] == 0.75

    def test_auc_perfect_and_nan_scores(self):
        m = binary_metrics([0, 0, 1, 1], [0, 0, 1, 1], scores=[0.1, np.nan, 0.8, 0.9])
        assert m["roc_auc"] == 1.0

    def test_wilson(self):
        lo, hi = wilson_ci(0, 100_000)
        assert lo == 0.0 and hi < 0.0001          # 0 FP sur 100 000 -> FPR < 0,01 % à 95 %
        assert all(math.isnan(x) for x in wilson_ci(0, 0))

    def test_metrics_by_family(self):
        df = pd.DataFrame({"family": ["a", "a", "b"], "label": [1, 1, 0], "pred": [1, 0, 1]})
        out = metrics_by(df, "family").set_index("family")
        assert out.loc["a", "recall"] == 0.5 and out.loc["b", "fpr"] == 1.0


# --------------------------------------------------------------------------- features autorisées
def _toy(n_benign=400, n_c2=40, seed=0):
    rng = np.random.default_rng(seed)
    n = n_benign + n_c2
    df = pd.DataFrame({
        "session_id": [f"s{i}" for i in range(n)],
        "label": [0] * n_benign + [1] * n_c2,
        "session_type": rng.choice(["dns", "https"], n),
        "split": rng.choice(["train", "val", "test"], n, p=[0.6, 0.2, 0.2]),
        "n_events": rng.integers(1, 50, n),
        "duration": rng.exponential(100, n),
        "orig_bytes": rng.exponential(1000, n), "resp_bytes": rng.exponential(5000, n),
        "orig_pkts": rng.integers(1, 100, n).astype(float), "resp_pkts": rng.integers(1, 100, n).astype(float),
        "n_conn_no_payload": 0.0,
        "dns_entropy_max": np.r_[rng.uniform(0, 2.5, n_benign), rng.uniform(4, 5, n_c2)],
        "dns_qname_len_max": np.r_[rng.integers(5, 30, n_benign), rng.integers(45, 60, n_c2)].astype(float),
        "dns_periodicity_score": np.r_[rng.uniform(0, 0.3, n_benign), rng.uniform(0.7, 1, n_c2)],
        "dns_iat_cv": np.r_[rng.uniform(0.8, 3, n_benign), rng.uniform(0, 0.1, n_c2)],
        "qname_truncated_share": 0.0,          # interdite : ne doit jamais être sélectionnée
    })
    df["group"] = df["session_type"].map({"dns": "dns", "https": "web"})
    return df


def test_feature_columns_exclude_forbidden():
    df = _toy()
    for g in ("dns", "web"):
        cols = p6_common.feature_columns(df, g)
        assert cols and not (set(cols) & p6_common.FORBIDDEN)
    assert "dns_entropy_max" not in p6_common.feature_columns(df, "web")   # contenu DNS : pas pour le web


def test_to_matrix_no_nan():
    df = _toy()
    df.loc[0, "dns_iat_cv"] = np.nan
    X = p6_common.to_matrix(df, p6_common.feature_columns(df, "dns"))
    assert np.isfinite(X).all()


# --------------------------------------------------------------------------- règles
def test_rules_expert_detect_obvious_c2():
    df = _toy()
    pred = predict(rule_scores(df), thresholds(rule_scores(df), df, "expert"))
    assert pred.loc[df.label == 1, "R_any"].mean() == 1.0
    assert pred.loc[df.label == 0, "R_any"].mean() < 0.05


def test_rule_regularity_needs_min_events():
    df = _toy().assign(n_events=3)
    assert not rule_scores(df)["R4_regularite"].notna().any()


def test_rules_calibrated_respect_fpr_on_train():
    df = _toy(n_benign=5000, n_c2=50)
    sc = rule_scores(df)
    pred = predict(sc, thresholds(sc, df, "calibrated"))
    benign_train = (df.split == "train") & (df.label == 0)
    for name, *_ in RULES:
        assert pred.loc[benign_train, name].mean() <= p6_common.TARGET_FPR / len(RULES) + 1e-12


def test_nan_feature_never_alerts():
    df = _toy().assign(dns_entropy_max=np.nan, dns_qname_len_max=np.nan,
                       dns_periodicity_score=np.nan, dns_iat_cv=np.nan)
    pred = predict(rule_scores(df), thresholds(rule_scores(df), df, "expert"))
    assert not pred["R_any"].any()


# --------------------------------------------------------------------------- Isolation Forest
def test_iforest_separates_toy_data():
    from p6_02_isolation_forest import score, train_iforest
    df = _toy(n_benign=2000, n_c2=100)
    b = train_iforest(df, "dns")
    test = df[(df.group == "dns") & (df.split == "test")]
    s = score(b, test)
    assert s[test.label.to_numpy() == 1].mean() > s[test.label.to_numpy() == 0].mean()
    assert b["n_train"] == ((df.group == "dns") & (df.split == "train") & (df.label == 0)).sum()


def test_iforest_trains_on_benign_only():
    from p6_02_isolation_forest import train_iforest
    df = _toy()
    b1 = train_iforest(df, "web")
    df2 = df.copy()
    df2.loc[(df2.label == 1), "orig_bytes"] *= 1000      # changer le C2 ne doit rien changer au modèle
    b2 = train_iforest(df2, "web")
    X = p6_common.to_matrix(df[df.label == 0], b1["columns"])
    assert np.allclose(b1["model"].score_samples(X), b2["model"].score_samples(X))


# --------------------------------------------------------------------------- pipeline P7
def test_pipeline_end_to_end_on_toy(tmp_path):
    from pipeline import C2Pipeline
    df = _toy()
    sessions = df.drop(columns=[c for c in df.columns if c.startswith("dns_")] + ["group"])
    sessions = sessions.assign(src_ip_hash="h", parent_domain="evil.com", server_name="x", ts_start=0.0)
    feats_path = tmp_path / "features_dns.parquet"
    df[["session_id"] + [c for c in df.columns if c.startswith("dns_")]].to_parquet(feats_path)

    pipe = C2Pipeline(models_dir=tmp_path)                 # pas de modèle -> règles seules
    feats = pipe.features(sessions, feats_path)
    assert len(feats) == len(sessions)                      # contrat : jointure gauche
    v = pipe.verdict(pipe.score(feats))
    assert set(v["alert"].unique()) <= {0, 1}
    assert v.loc[df.label == 1, "alert"].mean() == 1.0
    alerts = pipe.alerts(feats, v)
    assert len(alerts) == int(v["alert"].sum())
    assert {"session_id", "score", "detectors", "mitre_technique"} <= set(alerts[0])
    assert "load_models" in pipe.timings


def test_pipeline_live_mode_not_implemented_yet(tmp_path):
    from pipeline import C2Pipeline
    with pytest.raises(NotImplementedError):
        C2Pipeline(models_dir=tmp_path).run_pcap(tmp_path / "x.pcap")
