"""P7.3 - Alertes JSON + techniques MITRE (Salma).

build_alert(r) transforme une session alertée (une ligne de itertuples() dans pipeline.alerts())
en un dict conforme à alert_schema.json. map_mitre() associe le type de session aux techniques
MITRE ATT&CK.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pandas as pd

SCHEMA_VERSION = "1.0"

# Table MITRE : type de session -> techniques ATT&CK (tactique Command and Control)
_WEB = {"technique_id": "T1071.001", "name": "Application Layer Protocol: Web Protocols",
        "tactic": "Command and Control"}
MITRE = {
    "dns": [{"technique_id": "T1071.004", "name": "Application Layer Protocol: DNS",
             "tactic": "Command and Control"}],
    "https": [_WEB],
    "http": [_WEB],
}


def map_mitre(session_type: str) -> list[dict]:
    """Techniques MITRE pour un type de session (copie : l'appelant peut la modifier)."""
    return [dict(t) for t in MITRE.get(session_type, [])]


def severity(n_detectors: int) -> str:
    """low = 1 détecteur, medium = 2, high = 3 ou plus."""
    if n_detectors >= 3:
        return "high"
    if n_detectors == 2:
        return "medium"
    return "low"


def _none_if_na(x):
    """NaN / None de pandas -> None (null en JSON)."""
    return None if x is None or (not isinstance(x, str) and pd.isna(x)) else x


def build_alert(r) -> dict:
    """Une ligne alertée (r, de itertuples) -> une alerte au format de alert_schema.json."""
    session_id = str(r.session_id)
    ts = float(r.ts_start)
    destination = r.parent_domain if r.session_type == "dns" else r.server_name
    scores = {k: round(float(getattr(r, k)), 4) for k in ("iforest_score", "ae_score", "final_score")
              if hasattr(r, k) and not pd.isna(getattr(r, k))}
    if "final_score" in scores:          # fusion P6.6 (Oumaima), quand elle existera
        scores["final"] = scores.pop("final_score")
    detectors = [d for d in str(r.detectors).split(",") if d]

    return {
        "schema_version": SCHEMA_VERSION,
        "alert_id": "C2-" + hashlib.sha256(session_id.encode()).hexdigest()[:12],
        "session_id": session_id,
        "timestamp": ts,
        "time_utc": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "src_ip_hash": r.src_ip_hash,
        "destination": _none_if_na(destination),
        "session_type": r.session_type,
        "severity": severity(len(detectors)),
        "detectors": detectors,
        "scores": scores,
        "top_features": [],          # SHAP (Farah, P7.2) plus tard
        "mitre": map_mitre(r.session_type),
    }
