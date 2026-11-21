"""P7.3 - Tests du schéma JSON des alertes (Salma).

1. l'exemple examples/alert_example.json respecte le schéma ;
2. build_alert() produit des alertes valides (DNS, HTTPS, valeurs manquantes) ;
3. le mapping MITRE est correct ;
4. des alertes cassées sont bien REFUSÉES par le schéma.
"""
import copy
import json

import numpy as np
import pandas as pd
import pytest

jsonschema = pytest.importorskip("jsonschema")

from conftest import REPO  # noqa: E402  (ajoute aussi p7_app_demo au sys.path)
from alerts import build_alert, map_mitre, severity  # noqa: E402

P7 = REPO / "p7_app_demo"
SCHEMA = json.loads((P7 / "alert_schema.json").read_text(encoding="utf-8"))
EXAMPLE = json.loads((P7 / "examples" / "alert_example.json").read_text(encoding="utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)


def is_valid(alert: dict) -> bool:
    return VALIDATOR.is_valid(alert)


def row(**kw):
    """Une ligne alertée comme dans pipeline.alerts() (verdicts + colonnes de df)."""
    base = dict(session_id="s_001", alert=1, detectors="rules,iforest", rules_alert=True,
                iforest_score=0.7123, iforest_alert=True, session_type="dns",
                src_ip_hash="h_src", parent_domain="evil-domain.com", server_name=None,
                ts_start=1_600_000_000.0)
    base.update(kw)
    return next(pd.DataFrame([base]).itertuples())


# --------------------------------------------------------------------------- schéma + exemple
def test_schema_is_valid_json_schema():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


def test_example_matches_schema():
    VALIDATOR.validate(EXAMPLE)


# --------------------------------------------------------------------------- build_alert
def test_build_alert_dns_is_valid():
    a = build_alert(row())
    VALIDATOR.validate(a)
    assert a["destination"] == "evil-domain.com"
    assert a["detectors"] == ["rules", "iforest"]
    assert a["severity"] == "medium"
    assert a["time_utc"] == "2020-09-13T12:26:40Z"
    assert a["mitre"][0]["technique_id"] == "T1071.004"


def test_build_alert_https_uses_server_name():
    a = build_alert(row(session_type="https", parent_domain=None, server_name="cdn.example.org"))
    VALIDATOR.validate(a)
    assert a["destination"] == "cdn.example.org"
    assert a["mitre"][0]["technique_id"] == "T1071.001"


def test_build_alert_missing_values_become_null():
    a = build_alert(row(session_type="https", server_name=np.nan, iforest_score=np.nan))
    VALIDATOR.validate(a)
    assert a["destination"] is None
    assert "iforest_score" not in a["scores"]


def test_build_alert_is_json_serializable():
    json.dumps(build_alert(row(ae_score=np.float32(1.5))))


def test_alert_id_is_stable():
    assert build_alert(row())["alert_id"] == build_alert(row())["alert_id"]
    assert build_alert(row())["alert_id"] != build_alert(row(session_id="s_002"))["alert_id"]


def test_final_score_is_renamed():
    a = build_alert(row(final_score=0.95))
    assert a["scores"]["final"] == 0.95
    VALIDATOR.validate(a)


# --------------------------------------------------------------------------- MITRE + gravité
@pytest.mark.parametrize("stype, tid", [("dns", "T1071.004"), ("https", "T1071.001"), ("http", "T1071.001")])
def test_map_mitre(stype, tid):
    assert [t["technique_id"] for t in map_mitre(stype)] == [tid]


def test_map_mitre_unknown_type():
    assert map_mitre("ftp") == []


@pytest.mark.parametrize("n, sev", [(1, "low"), (2, "medium"), (3, "high"), (4, "high")])
def test_severity(n, sev):
    assert severity(n) == sev


# --------------------------------------------------------------------------- alertes cassées
def broken(**changes):
    a = copy.deepcopy(EXAMPLE)
    for k, v in changes.items():
        if v is KeyError:
            del a[k]
        else:
            a[k] = v
    return a


@pytest.mark.parametrize("alert", [
    broken(session_id=KeyError),                                    # champ obligatoire absent
    broken(session_type="ftp"),                                     # type inconnu
    broken(severity="critical"),                                    # gravité inconnue
    broken(timestamp="hier"),                                       # mauvais type
    broken(detectors=[]),                                           # aucun détecteur
    broken(mitre=[]),                                               # aucune technique
    broken(mitre=[{"technique_id": "1071", "name": "x", "tactic": "y"}]),  # mauvais format MITRE
    broken(top_features=[{"name": f"f{i}", "value": 1, "impact": 0.1} for i in range(4)]),  # > 3
    broken(alert_id="12345"),                                       # mauvais format d'id
    broken(sevrity="low"),                                          # champ inconnu (faute de frappe)
    broken(schema_version="2.0"),                                   # mauvaise version
], ids=["missing", "type", "severity", "timestamp", "no_detector", "no_mitre", "mitre_format",
        "too_many_features", "alert_id", "unknown_field", "version"])
def test_broken_alerts_are_rejected(alert):
    assert not is_valid(alert)
