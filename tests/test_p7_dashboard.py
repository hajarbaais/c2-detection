"""Tests du tableau de bord P7 (fonctions sans Streamlit : p7_app_demo/app/dashboard_lib.py)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p7_app_demo" / "app"))
import dashboard_lib as lib  # noqa: E402

PCAP_HEADER = b"\xd4\xc3\xb2\xa1" + b"\x00" * 20


@pytest.fixture(scope="module")
def demo():
    return lib.demo_result(n=200, seed=1)


def test_demo_has_pipeline_format(demo):
    assert set(demo) >= {"verdicts", "alerts", "timings", "data"}
    assert len(demo["verdicts"]) == len(demo["data"]) == 200
    assert {"session_id", "alert", "detectors"} <= set(demo["verdicts"].columns)
    assert len(demo["alerts"]) == int(demo["verdicts"]["alert"].sum()) > 0


def test_demo_is_reproducible():
    a, b = lib.demo_result(n=50, seed=3), lib.demo_result(n=50, seed=3)
    pd.testing.assert_frame_equal(a["verdicts"], b["verdicts"])


def test_demo_uses_only_reserved_domains(demo):
    dest = pd.concat([demo["data"]["parent_domain"], demo["data"]["server_name"]]).dropna()
    assert dest.str.endswith((".example.com", ".example.org", ".example.net", ".test")).all()


def test_sessions_table_alerts_first(demo):
    t = lib.sessions_table(demo)
    assert len(t) == 200
    n_alert = int(t["alerte"].sum())
    assert t["alerte"].iloc[:n_alert].all() and not t["alerte"].iloc[n_alert:].any()
    dns = t[t["type"] == "dns"]
    assert dns["destination"].notna().all()


def test_sessions_table_tolerates_missing_columns():
    res = {"verdicts": pd.DataFrame({"session_id": ["a", "b"], "alert": [1, 0]}),
           "data": pd.DataFrame({"session_id": ["a", "b"]})}
    t = lib.sessions_table(res)
    assert list(t["session_id"]) == ["a", "b"] and t["alerte"].tolist() == [True, False]


def test_row_style_red_only_for_alerts():
    assert "background-color" in lib.row_style(pd.Series({"alerte": True, "x": 1}))[0]
    assert lib.row_style(pd.Series({"alerte": False, "x": 1})) == ["", ""]


def test_severity_matches_p7_3():
    assert [lib.severity(n) for n in (0, 1, 2, 3, 5)] == ["", "low", "medium", "high", "high"]


def test_shap_frame_sorted_by_absolute_impact():
    alert = {"top_features": [{"name": "a", "value": 1, "impact": 0.1},
                              {"name": "b", "value": 2, "impact": -0.5},
                              {"name": "c", "value": 3, "impact": 0.3}]}
    df = lib.shap_frame(alert)
    assert df["name"].tolist() == ["b", "c", "a"]
    assert df["sens"].tolist() == ["vers normal", "vers C2", "vers C2"]


@pytest.mark.parametrize("alert", [None, {}, {"top_features": []}, {"top_features": [{"name": "x"}]}])
def test_shap_frame_empty_when_shap_missing(alert):
    assert lib.shap_frame(alert).empty


def test_timings_frame_order_and_latency():
    t = {"total": 1.2, "score": 0.5, "load_models": 2.0, "features": 0.6, "custom": 0.1}
    df = lib.timings_frame(t)
    assert df["étape"].iloc[0] == lib.STEP_LABELS["load_models"] and df["une_fois"].iloc[0]
    assert "total" not in df["étape"].tolist() and "custom" in df["étape"].tolist()
    assert lib.latency_ok(t) and not lib.latency_ok({"total": 6.0})
    assert lib.latency_ok({"load_models": 10.0, "score": 1.0})   # le chargement ne compte pas


def test_is_pcap():
    assert lib.is_pcap(PCAP_HEADER) and lib.is_pcap(b"\x0a\x0d\x0d\x0a....")
    assert not lib.is_pcap(b"MZ\x90\x00") and not lib.is_pcap(b"")


def test_save_upload_uses_generated_name(tmp_path):
    p = lib.save_upload(PCAP_HEADER, tmp_path)
    assert p.parent == tmp_path and p.name.startswith("upload_") and p.read_bytes() == PCAP_HEADER


def test_save_upload_rejects_non_pcap_and_big_files(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        lib.save_upload(b"#!/bin/sh\nrm -rf /", tmp_path)
    monkeypatch.setattr(lib, "MAX_UPLOAD_MB", 0)
    with pytest.raises(ValueError):
        lib.save_upload(PCAP_HEADER, tmp_path)
    assert not list(tmp_path.iterdir())
