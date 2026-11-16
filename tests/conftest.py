"""Outils communs aux tests : import des scripts (pas des packages) + mini-sortie P2 synthétique.

Lancer tous les tests depuis la racine du dépôt :   python -m pytest -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
P4_SCRIPTS = REPO / "p4_features_dns_time" / "scripts"
P6_SCRIPTS = REPO / "p6_models" / "scripts"
P6_EVAL = REPO / "p6_models" / "evaluation"
for p in (P6_SCRIPTS, P6_EVAL, REPO / "p7_app_demo"):
    sys.path.insert(0, str(p))


def load_script(path: Path):
    """Importe un script .py (ex. p4_01_entropy.py) comme module, sans exécuter main()."""
    name = "t_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [str(path)]           # les scripts P4 lisent sys.argv[1] au moment de l'import
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old_argv
    return mod


@pytest.fixture(scope="session")
def p4():
    """Les 5 scripts P4 de Hajar, importés comme modules : p4['01'].shannon_entropy, ..."""
    files = {"01": "p4_01_entropy", "02": "p4_02_shape", "03": "p4_03_iat",
             "04": "p4_04_periodicity", "05": "p4_05_windows"}
    return {k: load_script(P4_SCRIPTS / f"{v}.py") for k, v in files.items()}


# --------------------------------------------------------------------------- #
# Mini-sortie P2 synthétique (même schéma que sessions_pcap / events / events_pkts)
# --------------------------------------------------------------------------- #
BASE_TS = 1_600_000_000.0
EXFIL_CHARS = "abcdefghijklmnopqrstuvwxyz234567"   # alphabet base32


def _session(sid, stype, label, family, parent=None, key=None, window=0, file="f1.pcap", split="train"):
    return dict(session_id=sid, session_type=stype, label=label, family=family, parent_domain=parent,
                session_key=key or f"k_{sid}", window_idx=window, file=file, split=split,
                src_ip_hash="h_src", dst_ip_hash="h_dst", server_name=None, dst_port=53 if stype == "dns" else 443,
                ts_start=BASE_TS, duration=0.0, n_events=0, orig_bytes=np.nan, resp_bytes=np.nan,
                orig_pkts=np.nan, resp_pkts=np.nan, n_conn_no_payload=np.nan)


def make_p2(tmp_path: Path) -> Path:
    """Écrit sessions_pcap / events / events_pkts .parquet dans tmp_path et renvoie ce dossier.

    Sessions (ce qu'on attend des features) :
      beacon_dns  C2  : 60 requêtes toutes les 60 s +/- 3 s, même nom        -> CV bas, périodicité ~60 s
      exfil_dns   C2  : 30 requêtes TXT, sous-domaines base32 de 40 car., 10 NXDOMAIN -> entropie haute
      benign_dns  0   : 6 requêtes irrégulières sur google.com               -> entropie basse
      single_dns  0   : 1 seule requête                                      -> IAT / périodicité vides
      beacon_web  C2  : 40 paquets montants toutes les 30 s                   -> périodicité ~30 s
      empty_web   0   : aucune charge utile                                   -> absente des features
      winA0/winA1/winB0 : même session_key, fichiers différents (A: fenêtres 0 et 1 ; B: fenêtre 0)
    """
    rng = np.random.default_rng(0)
    sessions, events, pkts = [], [], []

    def add_dns(sid, queries, times, qtype=1, rcodes=None):
        for i, (q, t) in enumerate(zip(queries, times)):
            events.append(dict(session_id=sid, seq=i, ts=float(t), t_rel=float(t - times[0]),
                               event_type="dns_query", size=len(q) + 30, resp_size=100.0, query=q,
                               qtype=float(qtype), rcode=float(rcodes[i] if rcodes is not None else 0),
                               answers=1.0, rtt=0.01, qname_truncated=False))

    # beacon DNS
    sessions.append(_session("beacon_dns", "dns", 1, "beacon", "evil.com"))
    t = BASE_TS + np.cumsum(60 + rng.uniform(-3, 3, 60))
    add_dns("beacon_dns", ["cmd.evil.com"] * 60, t)
    # exfiltration DNS (TXT, base32, NXDOMAIN 10/30)
    sessions.append(_session("exfil_dns", "dns", 1, "dns2tcp", "tunnel.net"))
    subs = ["".join(rng.choice(list(EXFIL_CHARS), 40)) for _ in range(30)]
    add_dns("exfil_dns", [f"{s}.tunnel.net" for s in subs], BASE_TS + np.arange(30) * 0.5, qtype=16,
            rcodes=[3] * 10 + [0] * 20)
    # bénin DNS
    sessions.append(_session("benign_dns", "dns", 0, "benign_dns", "google.com"))
    add_dns("benign_dns", ["www.google.com", "mail.google.com", "www.google.com", "google.com",
                           "drive.google.com", "www.google.com"],
            BASE_TS + np.array([0, 1.5, 40, 41, 300, 900]))
    # une seule requête
    sessions.append(_session("single_dns", "dns", 0, "benign_dns", "example.com"))
    add_dns("single_dns", ["example.com"], [BASE_TS])
    # beacon web (paquets montants toutes les 30 s)
    sessions.append(_session("beacon_web", "https", 1, "trickbot"))
    for i in range(40):
        pkts.append(dict(session_id="beacon_web", seq=i, ts=BASE_TS + 30.0 * i, direction="up", size=200))
        pkts.append(dict(session_id="beacon_web", seq=i, ts=BASE_TS + 30.0 * i + 0.1, direction="down", size=900))
    # web sans charge utile
    sessions.append(_session("empty_web", "https", 0, "benign_https"))
    # même session_key dans deux pcaps différents
    for sid, win, f in [("winA0", 0, "A.pcap"), ("winA1", 1, "A.pcap"), ("winB0", 0, "B.pcap")]:
        sessions.append(_session(sid, "dns", 0, "benign_dns", "shared.org", key="KEY", window=win, file=f))
        add_dns(sid, ["www.shared.org"] * 3, BASE_TS + win * 3600 + np.array([0, 10, 25]))

    se = pd.DataFrame(sessions)
    ev = pd.DataFrame(events)
    se["n_events"] = se["session_id"].map(ev.groupby("session_id").size()).fillna(
        se["session_id"].map(pd.DataFrame(pkts).groupby("session_id").size())).fillna(1).astype(int)
    se.to_parquet(tmp_path / "sessions_pcap.parquet", index=False)
    ev.to_parquet(tmp_path / "events.parquet", index=False)
    pd.DataFrame(pkts).to_parquet(tmp_path / "events_pkts.parquet", index=False)
    return tmp_path


@pytest.fixture(scope="session")
def p4_run(p4, tmp_path_factory):
    """Exécute les 5 scripts P4 (main) sur la mini-sortie P2.

    Renvoie (features DataFrame indexé par session_id, dict des exceptions levées par script).
    Une exception n'arrête pas la chaîne : chaque script a déjà écrit son fichier avant sa
    vérification finale ; les tests dédiés signalent l'erreur.
    """
    p2 = make_p2(tmp_path_factory.mktemp("p2_output"))
    out = tmp_path_factory.mktemp("p4_output")
    errors = {}
    for k in ["01", "02", "03", "04", "05"]:
        mod = p4[k]
        mod.P2_OUT, mod.OUT_DIR = p2, out
        try:
            mod.main()
            errors[k] = None
        except Exception as e:     # noqa: BLE001
            errors[k] = e
    feats = pd.read_parquet(out / "features_dns.parquet").set_index("session_id")
    return feats, errors
