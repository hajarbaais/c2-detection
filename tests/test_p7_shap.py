"""Regression tests for the P7.2 alert explanation path."""
import json
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from pipeline import C2Pipeline
from p7_app_demo import shap_explanations


def test_top_features_signed_sorted_and_json_safe():
    columns = ["entropy", "regularity", "volume", "other"]
    background = pd.DataFrame([[0, 0, 1, 0]], columns=columns)
    rows = pd.DataFrame([[3, 2, np.nan, 1]], columns=columns)

    def score(frame):
        return 4 * frame.entropy - 3 * frame.regularity + 2 * frame.volume.fillna(0) + frame.other

    [top] = shap_explanations.explain_top_features(score, rows, background, columns)
    assert [item["name"] for item in top] == columns[:3]
    assert [item["impact"] for item in top] == pytest.approx([12, -6, -2])
    assert top[2]["value"] is None
    json.dumps(top, allow_nan=False)


def _sessions():
    return pd.DataFrame({
        "session_id": ["background", "normal", "rules", "both"],
        "group": ["dns"] * 4, "session_type": ["dns"] * 4,
        "split": ["train", "test", "test", "test"], "label": [0, 0, 1, 1],
        "src_ip_hash": ["h"] * 4, "parent_domain": ["example.test"] * 4,
        "server_name": [None] * 4, "ts_start": [0] * 4,
        "dns_entropy_max": [1, 1, 4, 4], "dns_qname_len_max": [10] * 4,
        "dns_periodicity_score": [0] * 4, "dns_iat_cv": [1] * 4, "n_events": [5] * 4,
        "signal": [0.0, 0.0, 0.0, 99.0],
    })


def test_pipeline_explains_only_alerts_with_the_triggering_scorer(tmp_path, monkeypatch):
    sessions = _sessions()
    pipe = C2Pipeline(models_dir=tmp_path, p5_dir=None, use_autoencoder=False)
    monkeypatch.setattr(pipe, "features", lambda frame: frame)
    forest = IsolationForest(n_estimators=12, random_state=0).fit(np.arange(20).reshape(-1, 1))
    # The edge at 0 can be as anomalous as 99. Use the center for the reference.
    sessions.loc[sessions["signal"].eq(0), "signal"] = 10
    normal, anomaly = -forest.score_samples([[10], [99]])
    assert normal < anomaly
    pipe.models = {"dns": {"model": forest, "columns": ["signal"], "threshold": (normal + anomaly) / 2}}
    seen = []
    explain = shap_explanations.explain_top_features

    def record(score, rows, reference, columns):
        seen.extend(rows["session_id"])
        assert reference["session_id"].tolist() == ["background"]
        return explain(score, rows, reference, columns)

    monkeypatch.setattr(shap_explanations, "explain_top_features", record)
    result = pipe._run(sessions, time.perf_counter())
    assert seen == ["both"]
    alerts = {alert["session_id"]: alert for alert in result["alerts"]}
    assert set(alerts) == {"both", "rules"}
    assert alerts["both"]["top_features"][0]["name"] == "signal"
    assert alerts["both"]["top_features"][0]["impact"] == pytest.approx(anomaly - normal)
    assert alerts["rules"]["top_features"] == []
    assert result["timings"]["shap"] >= 0


def test_pipeline_preserves_alert_without_training_reference(tmp_path, monkeypatch):
    pipe = C2Pipeline(models_dir=tmp_path, p5_dir=None, use_autoencoder=False)
    monkeypatch.setattr(pipe, "features", lambda frame: frame)
    forest = IsolationForest(n_estimators=12, random_state=0).fit(np.arange(20).reshape(-1, 1))
    pipe.models = {"dns": {"model": forest, "columns": ["signal"], "threshold": -1}}
    sessions = _sessions().iloc[[3]]
    with pytest.warns(UserWarning, match="no benign training reference"):
        [alert] = pipe._run(sessions, time.perf_counter())["alerts"]
    assert alert["top_features"] == []


def test_offline_limit_keeps_reference_and_explains_autoencoder(tmp_path, monkeypatch):
    sessions = _sessions()
    sessions.to_parquet(tmp_path / "sessions_pcap.parquet")
    pipe = C2Pipeline(models_dir=tmp_path, p5_dir=None, use_autoencoder=False)
    monkeypatch.setattr(pipe, "features", lambda frame: frame)

    def score(groups, frame):
        assert groups.keys() == {"dns"}
        errors = frame["dns_entropy_max"].pow(2)
        return pd.DataFrame({"ae_score": errors, "ae_alert": errors.gt(9)})

    pipe.autoencoder = {"groups": {"dns": {"cols": ["dns_entropy_max"]}}, "score": score}
    [alert] = pipe.run_from_p2_output(tmp_path, limit=1)["alerts"]
    assert alert["session_id"] == "rules"
    assert alert["scores"]["ae_score"] == 16
    assert alert["top_features"] == [{"name": "dns_entropy_max", "value": 4, "impact": 15}]
    [targeted] = pipe.run_from_p2_output(tmp_path, session_ids=["rules"])["alerts"]
    assert targeted["top_features"] == alert["top_features"]
    with pytest.raises(ValueError, match="Unknown session IDs"):
        pipe.run_from_p2_output(tmp_path, session_ids=["unknown"])


def test_pipeline_skips_shap_when_there_are_no_alerts(tmp_path, monkeypatch):
    pipe = C2Pipeline(models_dir=tmp_path, p5_dir=None, use_autoencoder=False)
    monkeypatch.setattr(pipe, "features", lambda frame: frame)

    def unexpected(*args):
        pytest.fail("SHAP must not run without alerts")

    monkeypatch.setattr("pipeline.explain_alerts", unexpected)
    assert pipe._run(_sessions().iloc[:2], time.perf_counter())["alerts"] == []


def test_shap_explains_saved_torch_autoencoder(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from p5_6a_autoencoder_proto import AutoEncoder, Prep

    sessions = _sessions()
    columns = ["dns_entropy_max"]
    prep = Prep().fit(sessions.iloc[[0]][columns])
    model = AutoEncoder(prep.n_out)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    torch.save({"groups": {"dns": {"state_dict": model.state_dict(), "dims": model.dims,
                                  "prep": prep.to_dict(), "threshold": 1.0}}},
               tmp_path / "autoencoder.pt")
    pipe = C2Pipeline(models_dir=tmp_path, p5_dir=None)
    monkeypatch.setattr(pipe, "features", lambda frame: frame)

    result = pipe._run(sessions, time.perf_counter())

    assert len(result["alerts"]) == 2
    for alert in result["alerts"]:
        assert alert["scores"]["ae_score"] == pytest.approx(9)
        assert alert["top_features"] == [{"name": "dns_entropy_max", "value": 4, "impact": 9}]


@pytest.mark.parametrize("name", ["lstm", "transformer"])
def test_shap_explains_saved_sequence_model(tmp_path, monkeypatch, name):
    torch = pytest.importorskip("torch")
    from p6_03_lstm import SeqLSTM
    from p6_05_transformer import SeqTransformer

    sessions = _sessions().assign(group="web", session_type="https")
    names = ["log_dt", "log_size", "log_resp_size", "direction", "is_dns", "is_nxdomain"]
    values = np.zeros((3, 3, 6), dtype=np.float32)
    values[:, :2, 3] = 1                     # packet, not fallback
    values[0, :2, 1] = 6                      # anomaly (id=both)
    values[:, 2] = 100                        # padding must not affect score or displayed value
    np.savez(tmp_path / "features_seq.npz", X=values, mask=np.tile([True, True, False], (3, 1)),
             session_id=["both", "background", "normal"], label=[1, 0, 0], split=["test", "train", "test"],
             session_type=["https"] * 3, feature_names=names)
    cls = SeqLSTM if name == "lstm" else SeqTransformer
    model = cls(n_feat=6, seq_len=3, mode="anomaly")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    path = tmp_path / f"{name}.pt"
    torch.save({"groups": {"web": {"state_dict": model.state_dict(), "dims": model.dims, "threshold": 1.0}},
                "mode": "anomaly", "subset": "packets", "features": names}, path)
    pipe = C2Pipeline(models_dir=tmp_path, p5_dir=tmp_path, **{f"{name}_path": path})
    monkeypatch.setattr(pipe, "features", lambda frame: frame)

    result = pipe._run(sessions, time.perf_counter())

    verdicts = result["verdicts"].set_index("session_id")
    assert pd.isna(verdicts.at["rules", f"{name}_score"])
    assert not verdicts.at["normal", "alert"]
    alert = next(alert for alert in result["alerts"] if alert["session_id"] == "both")
    assert alert["scores"][f"{name}_score"] == pytest.approx(37 / 6, abs=0.0001)
    [feature] = alert["top_features"]
    assert feature == {"name": "seq_log_size", "value": 6, "impact": pytest.approx(6)}
