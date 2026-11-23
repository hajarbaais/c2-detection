"""Tests : jitter réaliste (P6.4), LSTM (P6.3) et tableau comparatif LSTM / Transformer.

Données synthétiques au format P5 (features_seq.npz + seq_scaler.pkl), aucune donnée réelle nécessaire.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
joblib = pytest.importorskip("joblib")
from sklearn.preprocessing import StandardScaler  # noqa: E402

import p6_03_lstm as lstm  # noqa: E402
import p6_04_jitter as jit  # noqa: E402
import p6_compare_seq as cmp  # noqa: E402

MEAN, SCALE = 2.0, 1.5          # scaler de log1p(dt) fictif


def _seq(n=4, L=8, steps=(8, 5, 1, 3), seed=0):
    """X standardisé (n, L, 6) avec des dt connus, masque « préfixe » comme P5."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, L, 6), np.float32)
    mask = np.zeros((n, L), bool)
    for i, k in enumerate(steps):
        dt = rng.uniform(1, 120, k)
        dt[0] = 0.0                                  # premier pas d'une session : dt = 0 (P5)
        X[i, :k, 0] = (np.log1p(dt) - MEAN) / SCALE
        X[i, :k, 1:] = rng.normal(size=(k, 5))
        mask[i, :k] = True
    return torch.from_numpy(X), torch.from_numpy(mask)


def _dt(x):
    return torch.expm1(x[:, :, 0] * SCALE + MEAN)


# --------------------------------------------------------------------------- jitter
def test_jitter_zero_is_identity():
    x, m = _seq()
    assert torch.equal(jit.jitter_log_dt(x, m, 0.0, MEAN, SCALE), x)


@pytest.mark.parametrize("kind,lo,hi", [("sym", 0.5, 1.5), ("down", 0.5, 1.0)])
def test_jitter_ratio_in_bounds_and_only_dt_changes(kind, lo, hi):
    x, m = _seq()
    y = jit.jitter_log_dt(x, m, 0.5, MEAN, SCALE, kind, torch.Generator().manual_seed(0))
    d0, d1 = _dt(x), _dt(y)
    pos = m & (d0 > 1e-6)
    ratio = d1[pos] / d0[pos]
    assert ratio.min() >= lo - 1e-4 and ratio.max() <= hi + 1e-4
    assert ratio.std() > 0                                       # le facteur change d'un pas à l'autre
    assert torch.equal(y[:, :, 1:], x[:, :, 1:])                 # les 5 autres features ne bougent pas
    assert torch.equal(y[~m], x[~m])                             # padding intact
    assert torch.allclose(d1[m & (d0 <= 1e-6)], torch.zeros(1), atol=1e-4)   # dt = 0 reste 0


def test_jitter_per_session_level_and_reproducible():
    x, m = _seq()
    lv = torch.tensor([0.0, 0.5, 0.0, 0.3])
    y1 = jit.jitter_log_dt(x, m, lv, MEAN, SCALE, generator=torch.Generator().manual_seed(1))
    y2 = jit.jitter_log_dt(x, m, lv, MEAN, SCALE, generator=torch.Generator().manual_seed(1))
    assert torch.equal(y1, y2)
    assert torch.equal(y1[0], x[0]) and torch.equal(y1[2], x[2])
    assert not torch.equal(y1[1], x[1])


def test_load_dt_scaler(tmp_path):
    sc = StandardScaler().fit(np.c_[np.log1p([0, 10, 100.0]), [1, 2, 3.0], [0, 1, 2.0]])
    joblib.dump({"scaler": sc, "features": ["log_dt", "log_size", "log_resp_size"]}, tmp_path / "seq_scaler.pkl")
    mean, scale = jit.load_dt_scaler(tmp_path)
    assert mean == pytest.approx(sc.mean_[0]) and scale == pytest.approx(sc.scale_[0])


def test_merge_csv_keeps_other_groups(tmp_path):
    p = tmp_path / "r.csv"
    jit.merge_csv(p, pd.DataFrame({"group": ["web", "web"], "v": [1, 2]}))
    jit.merge_csv(p, pd.DataFrame({"group": ["dns"], "v": [3]}))
    out = jit.merge_csv(p, pd.DataFrame({"group": ["dns"], "v": [4]}))
    assert sorted(out["v"]) == [1, 2, 4]


# --------------------------------------------------------------------------- LSTM
@pytest.mark.parametrize("mode", ["supervised", "anomaly"])
def test_lstm_shapes_and_padding_ignored(mode):
    torch.manual_seed(0)
    x, m = _seq()
    model = lstm.SeqLSTM(6, 8, mode).eval()
    out = model(x, m)
    assert out.shape == ((4,) if mode == "supervised" else (4, 8, 6))
    x2 = x.clone()
    x2[~m] = 99.0                                    # valeurs bidon dans le padding
    with torch.no_grad():
        assert torch.allclose(model.encode(x, m), model.encode(x2, m), atol=1e-6)


def test_prefix_mask_check():
    _, m = _seq()
    assert lstm.check_prefix_mask(m)
    m2 = m.clone()
    m2[0, 0] = False
    assert not lstm.check_prefix_mask(m2)


# --------------------------------------------------------------------------- bout en bout
def _write_p5(d, n=600, L=16, seed=0):
    """features_seq.npz + seq_scaler.pkl : C2 = balise régulière ~60 s, bénin = dt irréguliers."""
    rng = np.random.default_rng(seed)
    label = (rng.random(n) < 0.3).astype(np.int8)
    split = rng.choice(["train", "val", "test"], n, p=[0.6, 0.2, 0.2]).astype(object)
    split[(label == 1) & (rng.random(n) < 0.1)] = "unseen"
    X = np.zeros((n, L, 6), np.float32)
    mask = np.zeros((n, L), bool)
    raw = []
    for i in range(n):
        k = rng.integers(4, L + 1)
        dt = rng.normal(60, 3, k) if label[i] else rng.exponential(30, k)
        dt = np.clip(dt, 0, None)
        dt[0] = 0
        X[i, :k, 0] = np.log1p(dt)
        X[i, :k, 1] = rng.normal(5 if label[i] else 6, 0.5, k)
        X[i, :k, 3] = rng.choice([-1.0, 1.0], k)
        mask[i, :k] = True
        raw.append(X[i, :k, :3])
    sc = StandardScaler().fit(np.concatenate(raw))
    X[:, :, :3] = np.where(mask[:, :, None], sc.transform(X[:, :, :3].reshape(-1, 3)).reshape(n, L, 3), 0)
    np.savez_compressed(d / "features_seq.npz", X=X, mask=mask, session_id=np.array([f"s{i}" for i in range(n)]),
                        label=label, split=split.astype(str), session_type=np.full(n, "https"),
                        feature_names=np.array(["log_dt", "log_size", "log_resp_size", "direction", "is_dns",
                                                "is_nxdomain"]))
    joblib.dump({"scaler": sc, "features": ["log_dt", "log_size", "log_resp_size"]}, d / "seq_scaler.pkl")


@pytest.mark.parametrize("mode", ["anomaly", "supervised"])
def test_end_to_end_train_save_reload_jitter_compare(tmp_path, monkeypatch, mode):
    p5, models, ev = tmp_path / "p5", tmp_path / "models", tmp_path / "eval"
    p5.mkdir()
    _write_p5(p5)
    for mod in (lstm, jit, cmp):
        monkeypatch.setattr(mod, "EVAL_DIR", ev, raising=False)
    monkeypatch.setattr(lstm, "MODELS_DIR", models)
    monkeypatch.setattr(cmp, "MD_OUT", tmp_path / "cmp.md")
    monkeypatch.setattr(cmp, "CSV_OUT", ev / "cmp.csv")

    pt = lstm.main(["--p5", str(p5), "--mode", mode, "--groups", "web", "--epochs", "3", "--patience", "2",
                    "--hidden", "16", "--no-mlflow", "--jitter-train", "0.5", "--out-name", "t"])
    tag = f"lstm_{mode}_all_jit50"
    met = pd.read_csv(ev / f"{tag}_metrics.csv")
    assert set(met.split) >= {"train", "val", "test", "unseen"}
    jr = pd.read_csv(ev / f"{tag}_jitter.csv")
    test = jr[jr.split == "test"].set_index("jitter")
    assert list(test.index) == [0.0, 0.1, 0.25, 0.5]
    assert test["fpr"].nunique() == 1                              # le bénin n'est jamais bruité
    # jitter 0 = métriques propres du test
    assert test.loc[0.0, "recall"] == pytest.approx(met.set_index("split").loc["test", "recall"])

    # rechargement : mêmes scores que pendant l'entraînement, et même protocole via la CLI P6.4
    loaded = lstm.load_lstm(pt)
    d = np.load(p5 / "features_seq.npz")
    s = lstm.score_lstm(loaded, d["X"], d["mask"], np.full(len(d["X"]), "web"))
    assert s["lstm_score"].notna().all()
    arch, ckpt, _ = jit.load_any(pt)
    assert arch == "lstm" and jit.ckpt_tag(arch, ckpt) == tag

    df = cmp.collect(ev)
    assert len(df) == 1 and df.iloc[0]["jitter_train"] == 0.5
    assert "LSTM + jitter-train 50 %" in cmp.to_markdown(df)
