"""P7 - Tableau de bord : fonctions sans Streamlit (testables avec pytest).

Le tableau de bord (app.py) affiche le résultat de C2Pipeline (pipeline.py) :
    {"verdicts": DataFrame, "alerts": list[dict], "timings": dict[str, float], "data": DataFrame}

Ce module ne recode AUCUNE détection : il met en forme ce résultat (tableau, SHAP, chronomètre),
contrôle les fichiers déposés et fournit des données SIMULÉES pour le mode démo.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

LATENCY_TARGET_S = 5.0          # cahier des charges : alerte en moins de 5 s
MAX_UPLOAD_MB = 200             # aligné sur la limite par défaut de Streamlit

# Premiers octets d'un fichier pcap (4 ordres d'octets / précisions) ou pcapng
PCAP_MAGICS = (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d",
               b"\x0a\x0d\x0d\x0a")

STEP_LABELS = {
    "load_models": "Chargement des modèles (une fois)",
    "load_sessions": "Lecture des sessions",
    "extract": "Extraction (PCAP → journaux)",
    "sessionize": "Sessionisation",
    "features": "Features (P4 + P5)",
    "score": "Scores des détecteurs (P6)",
    "verdict": "Verdict",
    "alerts": "Alertes JSON",
}

SCORE_COLUMNS = ["iforest_score", "ae_score", "lstm_score", "final_score"]


# ---------------------------------------------------------------- fichiers déposés
def is_pcap(head: bytes) -> bool:
    """Vrai si les premiers octets sont ceux d'un pcap ou d'un pcapng."""
    return head[:4] in PCAP_MAGICS


def save_upload(data: bytes, workdir: Path) -> Path:
    """Écrit le fichier déposé sous un nom GÉNÉRÉ (uuid).

    Le nom donné par l'utilisateur n'est jamais réutilisé : ni dans un chemin, ni dans une commande
    système (règle de docs/docker.md). Le contenu n'est jamais exécuté, seulement lu par l'extraction.
    """
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"fichier trop gros (> {MAX_UPLOAD_MB} Mo)")
    if not is_pcap(data):
        raise ValueError("ce fichier n'est pas un pcap / pcapng (signature inconnue)")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"upload_{uuid.uuid4().hex}.pcap"
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------- mise en forme
def severity(n_detectors: int) -> str:
    """Même règle que P7.3 (alerts.py) : 1 détecteur = low, 2 = medium, 3 ou plus = high."""
    if n_detectors >= 3:
        return "high"
    if n_detectors == 2:
        return "medium"
    return "low" if n_detectors == 1 else ""


def sessions_table(res: dict) -> pd.DataFrame:
    """Une ligne par session : identifiant, type, destination, détecteurs, scores, alerte."""
    v = res["verdicts"].reset_index(drop=True)
    d = res["data"].reset_index(drop=True)
    out = pd.DataFrame({"session_id": v["session_id"].astype(str)})
    stype = d["session_type"] if "session_type" in d else pd.Series([""] * len(d))
    out["type"] = stype.to_numpy()
    dns_dest = d["parent_domain"] if "parent_domain" in d else pd.Series([None] * len(d))
    web_dest = d["server_name"] if "server_name" in d else pd.Series([None] * len(d))
    out["destination"] = np.where(stype.to_numpy() == "dns", dns_dest.to_numpy(), web_dest.to_numpy())
    out["alerte"] = v["alert"].astype(int).astype(bool).to_numpy()
    dets = v["detectors"].fillna("").astype(str) if "detectors" in v else pd.Series([""] * len(v))
    out["détecteurs"] = dets.to_numpy()
    out["sévérité"] = [severity(len([x for x in s.split(",") if x])) for s in dets]
    for c in SCORE_COLUMNS:
        if c in v:
            out[c] = v[c].astype(float).round(4).to_numpy()
    # Les alertes d'abord, puis le score le plus élevé
    sort_cols = ["alerte"] + [c for c in SCORE_COLUMNS if c in out]
    return out.sort_values(sort_cols, ascending=False, na_position="last").reset_index(drop=True)


def row_style(row: pd.Series) -> list[str]:
    """Styler pandas : ligne en rouge si la session est en alerte."""
    css = "background-color: #fde2e1; color: #8a1c1c; font-weight: 600" if row.get("alerte") else ""
    return [css] * len(row)


def alerts_by_session(alerts: list[dict]) -> dict[str, dict]:
    return {str(a.get("session_id")): a for a in alerts or []}


def shap_frame(alert: dict | None) -> pd.DataFrame:
    """top_features d'une alerte (format P7.3 : name, value, impact) -> DataFrame trié par |impact|.

    impact > 0 : pousse vers « C2 » ; impact < 0 : pousse vers « normal ».
    Vide tant que SHAP (P7.2, Farah) ne remplit pas top_features.
    """
    feats = (alert or {}).get("top_features") or []
    rows = [f for f in feats if isinstance(f, dict) and "name" in f and "impact" in f]
    df = pd.DataFrame(rows, columns=["name", "value", "impact"])
    if df.empty:
        return df
    df["impact"] = df["impact"].astype(float)
    df["sens"] = np.where(df["impact"] >= 0, "vers C2", "vers normal")
    return df.reindex(df["impact"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def timings_frame(timings: dict[str, float]) -> pd.DataFrame:
    """Chronométrage par étape (hors « total »), dans l'ordre du pipeline."""
    order = list(STEP_LABELS)
    steps = [k for k in order if k in timings] + [k for k in timings if k not in order and k != "total"]
    return pd.DataFrame({
        "étape": [STEP_LABELS.get(k, k) for k in steps],
        "secondes": [round(float(timings[k]), 3) for k in steps],
        "une_fois": [k == "load_models" for k in steps],
    })


def latency_ok(timings: dict[str, float], target: float = LATENCY_TARGET_S) -> bool:
    """Le total (hors chargement des modèles, fait une seule fois) respecte-t-il l'objectif ?"""
    total = timings.get("total")
    if total is None:
        total = sum(t for k, t in timings.items() if k != "load_models")
    return float(total) < target


# ---------------------------------------------------------------- données simulées (mode démo)
_DNS_FEATURES = [("dns_entropy_mean", 3.0, 4.4), ("dns_qname_len_mean", 18, 55), ("dns_iat_cv", 1.2, 0.05),
                 ("dns_periodicity_score", 0.1, 0.9), ("dns_unique_ratio", 0.3, 0.98),
                 ("dns_pct_txt", 0.0, 0.6)]
_WEB_FEATURES = [("p5_ja3_rarity", 0.05, 0.95), ("p5_cert_self_signed", 0, 1),
                 ("p5_log_domain_reputation", 9.0, 1.5), ("p5_conn_rate", 0.2, 3.5),
                 ("p5_avg_pkt_size", 900, 180), ("duration", 30, 3600)]


def demo_result(n: int = 300, seed: int = 0, alert_rate: float = 0.08) -> dict:
    """Résultat au MÊME format que C2Pipeline._run(), avec des données inventées.

    Sert à développer et présenter l'interface sans les données ni les modèles (Drive).
    Les domaines sont en .example / .test (réservés, aucun vrai domaine) et les IP sont des hachages factices.
    """
    rng = np.random.default_rng(seed)
    stype = rng.choice(["dns", "https", "http"], size=n, p=[0.55, 0.38, 0.07])
    is_c2 = rng.random(n) < alert_rate
    sid = [f"{t}_{hashlib.sha256(f'{seed}-{i}'.encode()).hexdigest()[:8]}_w{i % 24}" for i, t in enumerate(stype)]
    rand_label = lambda k: "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), k))  # noqa: E731
    benign = ["cdn.example.com", "mail.example.org", "api.example.net", "static.example.com", "updates.example.org"]
    dest = [f"{rand_label(10)}.c2-demo.test" if c else str(rng.choice(benign)) for c in is_c2]

    data = pd.DataFrame({
        "session_id": sid,
        "session_type": stype,
        "src_ip_hash": [hashlib.sha256(f"ip-{seed}-{i % 40}".encode()).hexdigest()[:16] for i in range(n)],
        "parent_domain": np.where(stype == "dns", dest, None),
        "server_name": np.where(stype != "dns", dest, None),
        "ts_start": 1_763_000_000 + np.sort(rng.uniform(0, 3600, n)),
    })
    if_score = np.where(is_c2, rng.normal(0.70, 0.06, n), rng.normal(0.45, 0.05, n)).round(4)
    ae_score = np.where(is_c2, rng.normal(3.2, 0.6, n), rng.normal(0.8, 0.3, n)).clip(0).round(4)
    rules = is_c2 & (rng.random(n) < 0.5)
    iforest = is_c2 & (rng.random(n) < 0.8)
    ae = (is_c2 & (rng.random(n) < 0.9)) | (rng.random(n) < 0.01)     # quelques fausses alertes
    verdicts = pd.DataFrame({"session_id": sid, "rules_alert": rules, "iforest_score": if_score,
                             "iforest_alert": iforest, "ae_score": ae_score, "ae_alert": ae})
    alert_cols = ["rules_alert", "iforest_alert", "ae_alert"]
    verdicts["alert"] = verdicts[alert_cols].any(axis=1).astype("int8")
    verdicts["detectors"] = verdicts[alert_cols].apply(
        lambda r: ",".join(c[:-6] for c in alert_cols if r[c]), axis=1)

    alerts = []
    for i in np.flatnonzero(verdicts["alert"].to_numpy()):
        spec = _DNS_FEATURES if stype[i] == "dns" else _WEB_FEATURES
        top = []
        for name, normal, c2 in spec:
            val = c2 if is_c2[i] else normal
            val = float(round(val * rng.uniform(0.85, 1.15), 3))
            impact = float(round(rng.uniform(0.05, 0.4) * (1 if is_c2[i] else rng.choice([-1, 1])), 3))
            top.append({"name": name, "value": val, "impact": impact})
        dets = [d for d in verdicts.at[i, "detectors"].split(",") if d]
        alerts.append({
            "schema_version": "1.0",
            "alert_id": "C2-" + hashlib.sha256(sid[i].encode()).hexdigest()[:12],
            "session_id": sid[i], "timestamp": float(data.at[i, "ts_start"]),
            "src_ip_hash": data.at[i, "src_ip_hash"], "destination": dest[i], "session_type": stype[i],
            "severity": severity(len(dets)), "detectors": dets,
            "scores": {"iforest_score": float(if_score[i]), "ae_score": float(ae_score[i])},
            "top_features": sorted(top, key=lambda f: -abs(f["impact"]))[:5],
            "mitre": [{"technique_id": "T1071.004" if stype[i] == "dns" else "T1071.001",
                       "name": "Application Layer Protocol: " + ("DNS" if stype[i] == "dns" else "Web Protocols"),
                       "tactic": "Command and Control"}],
        })
    # Ordres de grandeur mesurés dans p7_app_demo/README.md (CPU), mis à l'échelle sur n sessions
    k = n / 184_979
    timings = {"load_models": 2.0, "load_sessions": 0.02 + 0.4 * k, "features": 0.05 + 0.8 * k,
               "score": 0.05 + 0.6 * k, "verdict": 0.01 + 0.05 * k, "alerts": 0.005}
    timings["total"] = sum(t for s, t in timings.items() if s != "load_models")
    data["label"] = is_c2.astype("int8")
    return {"verdicts": verdicts, "alerts": alerts, "timings": timings, "data": data, "simulated": True}
