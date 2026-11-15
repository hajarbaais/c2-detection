"""Tests : features TLS de P5 recalculées sur le bénin (p6_tls.py) + autoencodeur dans le pipeline P7."""
import numpy as np
import pandas as pd
import pytest

import p6_common
from p6_tls import P5_COLS, P5_FEATS, apply_tls_reference, fit_tls_reference


def _web(n=300, seed=0):
    """Sessions web + quelques DNS, avec JA3/JA4/domaines et les 14 colonnes p5_ déjà jointes."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "session_id": [f"w{i}" for i in range(n)],
        "label": (rng.random(n) < 0.2).astype(int),
        "split": rng.choice(["train", "val", "test"], n, p=[0.6, 0.2, 0.2]),
        "session_type": rng.choice(["https", "https", "https", "dns"], n),
        "ja3": rng.choice(["a", "b", "c", None], n),
        "ja4": rng.choice(["x", "y", None], n),
        "server_name": rng.choice(["google.com", "evil.net", None], n),
        "dst_ip_hash": rng.choice(["d1", "d2"], n),
        "src_ip_hash": rng.choice(["h1", "h2", "h3"], n),
    })
    df["group"] = df["session_type"].map({"dns": "dns", "https": "web"})
    for c in ["dns_entropy_max", "dns_qname_len_max", "dns_periodicity_score", "dns_iat_cv"]:
        df[c] = np.nan                                  # colonnes lues par les règles P6.1
    for c in P5_FEATS:
        df[c] = rng.normal(size=n)
    return df


def test_reference_uses_benign_train_web_only():
    df = _web()
    ref = fit_tls_reference(df)
    assert ref["n_ref"] == ((df.split == "train") & (df.label == 0) & (df.group == "web")).sum()
    df2 = df.copy()
    c2_or_other = ~((df2.split == "train") & (df2.label == 0) & (df2.group == "web"))
    df2.loc[c2_or_other, "ja3"] = "zzz"                 # modifier C2 / val / test ne change pas la référence
    assert fit_tls_reference(df2) == ref


def test_rarity_unseen_is_max_and_missing_is_zero():
    df = _web()
    ref = fit_tls_reference(df)
    new = df.head(3).assign(ja3=["jamais_vu", None, "a"], group="web")
    out = apply_tls_reference(new, ref)
    assert out["p5_ja3_rarity"].iloc[0] == pytest.approx(np.log(ref["n_ref"] + 1))
    assert out["p5_ja3_rarity"].iloc[1] == 0.0
    assert 0 < out["p5_ja3_rarity"].iloc[2] < out["p5_ja3_rarity"].iloc[0]


def test_same_values_as_autoencoder_training():
    """Le pipeline doit fournir à l'autoencodeur EXACTEMENT les valeurs vues à l'entraînement (P5.6b)."""
    pytest.importorskip("torch")
    from p5_6b_autoencoder_train import recompute_p5_on_benign
    df = _web()
    a = recompute_p5_on_benign(df)
    b = apply_tls_reference(df, fit_tls_reference(df))
    web = df.group == "web"
    for c in ["p5_ja3_rarity", "p5_ja4_rarity", "p5_log_domain_reputation"]:
        assert np.allclose(a.loc[web, c], b.loc[web, c])


def test_feature_columns_with_p5():
    df = _web()
    cols = p6_common.feature_columns(df, "web", with_p5=True)
    assert set(P5_FEATS) <= set(cols) and not set(p6_common.SESSION_NUMERIC) & set(cols)
    with pytest.raises(AssertionError):
        p6_common.feature_columns(df.drop(columns=["p5_has_ja3"]), "web", with_p5=True)
    assert len(P5_COLS) == 14


def _save_tiny_autoencoder(df, path):
    """autoencoder.pt au format de P5.6b (non entraîné : on teste le branchement, pas la qualité)."""
    torch = pytest.importorskip("torch")
    from p5_6a_autoencoder_proto import AutoEncoder, Prep
    groups = {}
    for g, cols in {"dns": ["n_events", "duration"], "web": P5_FEATS}.items():
        prep = Prep().fit(df.loc[df.group == g, cols])
        model = AutoEncoder(prep.n_out)
        groups[g] = {"state_dict": model.state_dict(), "dims": model.dims, "prep": prep.to_dict(),
                     "threshold": 1.0, "history": []}
    torch.save({"groups": groups}, path)


def test_pipeline_scores_with_autoencoder(tmp_path):
    from pipeline import C2Pipeline
    df = _web().assign(n_events=3, duration=10.0, parent_domain="x", ts_start=0.0)
    _save_tiny_autoencoder(df, tmp_path / "autoencoder.pt")
    pipe = C2Pipeline(models_dir=tmp_path, p5_dir=None)
    assert pipe.autoencoder is not None
    sc = pipe.score(df)
    assert sc["ae_score"].notna().all()                 # dns et web scorés
    assert sc["ae_alert"].dtype == bool
    v = pipe.verdict(sc)
    assert "ae_alert" in v.columns and set(v["alert"].unique()) <= {0, 1}
    alerts = pipe.alerts(df.set_index(v.index), v.assign(alert=1))
    assert "ae_score" in alerts[0]["scores"]


def test_pipeline_skips_autoencoder_group_without_p5(tmp_path):
    from pipeline import C2Pipeline
    df = _web().assign(n_events=3, duration=10.0)
    _save_tiny_autoencoder(df, tmp_path / "autoencoder.pt")
    sc = C2Pipeline(models_dir=tmp_path, p5_dir=None).score(df.drop(columns=P5_FEATS))
    assert sc.loc[df.group == "dns", "ae_score"].notna().all()
    assert sc.loc[df.group == "web", "ae_score"].isna().all()   # colonnes P5 absentes -> ignoré, pas d'erreur
    assert not sc.loc[df.group == "web", "ae_alert"].any()
