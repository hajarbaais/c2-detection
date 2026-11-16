"""P7.1 - Squelette du pipeline bout-en-bout : PCAP -> sessions -> features -> scores -> verdict.

Objectif du squelette (sprint 2) : figer les INTERFACES entre les lots pour que chacun branche
son morceau sans casser les autres. Chaque étape est une méthode avec une entrée / sortie fixe :

    extract(pcap)            -> dossier de journaux (conn/dns/ssl/pkts)         [P2, tshark]
    sessionize(logs)         -> sessions (DataFrame) + events + events_pkts      [P2]
    features(sessions, ...)  -> 1 ligne par session_id                           [P4 + P5]
    score(features)          -> 1 colonne de score par modèle                    [P6]
    verdict(scores)          -> alert (0/1) + score final                        [P6 fusion]
    alerts(verdicts)         -> liste de dicts (format JSON SIEM)                [P7.3]

Les modèles sont chargés UNE SEULE FOIS dans __init__ (contrainte < 5 s par alerte).
Chaque étape est chronométrée (self.timings) pour suivre la latence dès maintenant.

État actuel :
  OK      score() : règles P6.1 + Isolation Forest P6.2 + autoencodeur P5.6 (Oumaima)
          (chaque détecteur est utilisé si son fichier existe dans p6_models/models/)
  OK      features() : P4 (features_dns.parquet) + P5 (features_tls.parquet, groupe web) ; rareté JA3/JA4
          et réputation recalculées avec la référence du bénin du train (models/tls_reference.joblib)
  OK      run_from_p2_output() : rejoue le pipeline sur des sorties P2/P4/P5 déjà calculées
  À FAIRE extract / sessionize sur un PCAP live (réutiliser p2_01_extract / p2_02_sessions)
  À FAIRE features() en mémoire : Hajar/Imane exposent une fonction compute(sessions, events, pkts)
  À FAIRE LSTM (Wassima S3), fusion (Oumaima S3), SHAP (Farah), JSON + MITRE (Salma)

Usage :
  python -I p7_app_demo/pipeline.py --p2 DOSSIER_OUTPUT_P2 [--p5 DOSSIER_P5_OUTPUT] [--limit 5000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "p6_models" / "scripts"))
from p6_common import GROUPS, MODELS_DIR, P4_FEATURES, p2_dir, to_matrix  # noqa: E402
from p6_01_rule_baselines import predict as rules_predict  # noqa: E402
from p6_01_rule_baselines import rule_scores, RULES  # noqa: E402
from p6_tls import apply_tls_reference, load_reference, merge_p5  # noqa: E402

P5_DIR = REPO / "p5_features_tls_seq" / "output"   # features_tls.parquet d'Imane (mode hors-ligne)
AE_FILE = "autoencoder.pt"


class C2Pipeline:
    """Chaîne de détection. Instancier une fois, appeler run_*() pour chaque capture."""

    def __init__(self, models_dir: Path = MODELS_DIR, use_iforest: bool = True,
                 use_autoencoder: bool = True, p5_dir: Path | None = P5_DIR):
        self.timings: dict[str, float] = {}
        self.p5_dir = Path(p5_dir) if p5_dir else None
        with self._timer("load_models"):
            self.models = self._load_models(models_dir) if use_iforest else {}
            self.rule_thresholds = self._load_rule_thresholds(models_dir)
            self.tls_reference = load_reference(models_dir)
            self.autoencoder = self._load_autoencoder(models_dir) if use_autoencoder else None
            # TODO S3 : self.sequence_model = torch.load(...)  (LSTM, Wassima)

    # ------------------------------------------------------------------ utilitaires
    @contextmanager
    def _timer(self, step: str):
        t0 = time.perf_counter()
        yield
        self.timings[step] = time.perf_counter() - t0

    @staticmethod
    def _load_rule_thresholds(models_dir: Path) -> dict:
        """Seuils calibrés de P6.1 (rules_thresholds.json) sinon seuils experts."""
        path = Path(models_dir) / "rules_thresholds.json"
        if path.exists():
            return json.loads(path.read_text())
        return {name: thr for name, _, _, thr in RULES}

    @staticmethod
    def _load_models(models_dir: Path) -> dict:
        import joblib
        models = {}
        for group in GROUPS:
            path = Path(models_dir) / f"iforest_{group}.joblib"
            if path.exists():
                models[group] = joblib.load(path)
        if not models:
            print(f"[pipeline] aucun modèle dans {models_dir} : lancer p6_02_isolation_forest.py "
                  "(seules les règles seront utilisées)")
        return models

    @staticmethod
    def _load_autoencoder(models_dir: Path):
        """autoencoder.pt (Oumaima, P5.6b) : modèle + prétraitement + seuil de chaque groupe."""
        path = Path(models_dir) / AE_FILE
        if not path.exists():
            print(f"[pipeline] {path.name} absent (à récupérer sur le Drive) : autoencodeur désactivé")
            return None
        try:
            from p5_6b_autoencoder_train import load_autoencoder, score_autoencoder
        except ImportError as e:   # torch non installé
            print(f"[pipeline] autoencodeur désactivé ({e})")
            return None
        return {"groups": load_autoencoder(path), "score": score_autoencoder}

    @staticmethod
    def _usable(df: pd.DataFrame, cols: list[str], name: str, group: str) -> bool:
        """Un détecteur n'est appliqué que si toutes ses colonnes sont présentes (sinon : NaN + message)."""
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[pipeline] {name}/{group} ignoré : colonnes absentes {missing[:3]}"
                  f"{'...' if len(missing) > 3 else ''} (features P5 non chargées ?)")
        return not missing

    # ------------------------------------------------------------------ étapes
    def extract(self, pcap: Path) -> Path:
        """PCAP -> journaux. TODO (S3/S4) : appeler l'extraction tshark de P2 sur un seul fichier."""
        raise NotImplementedError("extraction live : brancher p2_ingestion/scripts/p2_01_extract.py")

    def sessionize(self, logs_dir: Path):
        """Journaux -> (sessions, events, events_pkts). TODO : brancher p2_02_sessions.py."""
        raise NotImplementedError("sessionisation live : brancher p2_ingestion/scripts/p2_02_sessions.py")

    def features(self, sessions: pd.DataFrame, features_path: Path = P4_FEATURES) -> pd.DataFrame:
        """Sessions -> features. Pour l'instant : relit features_dns.parquet produit par les scripts P4.

        Contrat : renvoie EXACTEMENT les lignes de `sessions` (jointure gauche sur session_id),
        avec une colonne `group` (dns / web). Les sessions sans features gardent des NaN.
        """
        feats = pd.read_parquet(features_path) if Path(features_path).exists() else pd.DataFrame(
            {"session_id": pd.Series(dtype=str)})
        df = sessions.merge(feats, on="session_id", how="left")
        df["group"] = df["session_type"].map({t: g for g, ts in GROUPS.items() for t in ts})
        # features P5 (TLS, Imane) : groupe web seulement, puis référence « bénin du train »
        if self.p5_dir and (self.p5_dir / "features_tls.parquet").exists():
            df = merge_p5(df, self.p5_dir)
            if self.tls_reference is not None:
                df = apply_tls_reference(df, self.tls_reference)
            else:
                print("[pipeline] tls_reference.joblib absent : rareté/réputation P5 d'origine "
                      "(lancer p6_02_isolation_forest.py --p5 ...)")
        return df

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features -> scores (1 colonne par détecteur, élevé = suspect) + alertes par détecteur."""
        out = pd.DataFrame({"session_id": df["session_id"].to_numpy()}, index=df.index)
        rs = rule_scores(df)
        out["rules_alert"] = rules_predict(rs, self.rule_thresholds)["R_any"].to_numpy()
        out["iforest_score"] = np.nan
        out["iforest_alert"] = False
        for group, b in self.models.items():
            m = (df["group"] == group).to_numpy()
            if m.any() and self._usable(df, b["columns"], "iforest", group):
                s = -b["model"].score_samples(to_matrix(df[m], b["columns"]))
                out.loc[m, "iforest_score"] = s
                out.loc[m, "iforest_alert"] = s > b["threshold"]
        if self.autoencoder:
            groups = {g: b for g, b in self.autoencoder["groups"].items()
                      if self._usable(df, b["cols"], "autoencoder", g)}
            ae = self.autoencoder["score"](groups, df)
            out["ae_score"] = ae["ae_score"].to_numpy()
            out["ae_alert"] = ae["ae_alert"].astype(bool).to_numpy()
        # TODO S3 : out["lstm_score"], out["lstm_alert"]
        return out

    def verdict(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Scores -> verdict final. Provisoire : alerte si AU MOINS un détecteur alerte.

        TODO S3 (Oumaima, P6.6) : remplacer par la fusion calibrée AE + séquentiel.
        """
        v = scores.copy()
        alert_cols = [c for c in v.columns if c.endswith("_alert")]
        v["alert"] = v[alert_cols].any(axis=1).astype("int8")
        v["detectors"] = v[alert_cols].apply(lambda r: ",".join(c[:-6] for c in alert_cols if r[c]), axis=1)
        return v

    @staticmethod
    def alerts(df: pd.DataFrame, verdicts: pd.DataFrame) -> list[dict]:
        """Verdicts -> alertes JSON. Format PROVISOIRE ; schéma final + MITRE : Salma (P7.3)."""
        alerted = verdicts[verdicts["alert"] == 1].join(
            df[["session_type", "src_ip_hash", "parent_domain", "server_name", "ts_start"]])
        return [{
            "session_id": r.session_id,
            "timestamp": float(r.ts_start),
            "src_ip_hash": r.src_ip_hash,
            "destination": r.parent_domain if r.session_type == "dns" else r.server_name,
            "session_type": r.session_type,
            "scores": {k: round(float(getattr(r, k)), 4) for k in ("iforest_score", "ae_score")
                       if hasattr(r, k) and not pd.isna(getattr(r, k))},
            "detectors": r.detectors,
            "top_features": [],          # TODO S3 : SHAP (Farah, P7.2)
            "mitre_technique": None,     # TODO S3 : Salma (P7.3) — ex. T1071.004 (DNS), T1071.001 (web)
        } for r in alerted.itertuples()]

    # ------------------------------------------------------------------ points d'entrée
    def run_from_p2_output(self, p2: Path, limit: int | None = None) -> dict:
        """Rejoue le pipeline à partir des sorties P2 + P4 déjà calculées (mode hors-ligne)."""
        t0 = time.perf_counter()
        with self._timer("load_sessions"):
            sessions = pd.read_parquet(Path(p2) / "sessions_pcap.parquet")
            if limit:
                sessions = sessions.sample(min(limit, len(sessions)), random_state=0)
            sessions = sessions.reset_index(drop=True)
        return self._run(sessions, t0)

    def run_pcap(self, pcap: Path) -> dict:
        """Mode live (démo Streamlit). Fonctionnera quand extract/sessionize/features seront branchés."""
        t0 = time.perf_counter()
        with self._timer("extract"):
            logs = self.extract(pcap)
        with self._timer("sessionize"):
            sessions, _events, _pkts = self.sessionize(logs)
        return self._run(sessions, t0)

    def _run(self, sessions: pd.DataFrame, t0: float) -> dict:
        with self._timer("features"):
            df = self.features(sessions)
        with self._timer("score"):
            scores = self.score(df)
        with self._timer("verdict"):
            verdicts = self.verdict(scores)
        with self._timer("alerts"):
            alerts = self.alerts(df, verdicts)
        self.timings["total"] = time.perf_counter() - t0
        return {"verdicts": verdicts, "alerts": alerts, "timings": dict(self.timings), "data": df}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2", help="dossier output de P2")
    ap.add_argument("--p5", help="dossier p5_output (features_tls.parquet) ; défaut : "
                                 "p5_features_tls_seq/output")
    ap.add_argument("--limit", type=int, default=None, help="nb de sessions tirées au hasard (test rapide)")
    args = ap.parse_args()

    pipe = C2Pipeline(p5_dir=Path(args.p5) if args.p5 else P5_DIR)
    res = pipe.run_from_p2_output(p2_dir(args.p2), limit=args.limit)
    v, df = res["verdicts"], res["data"]
    n_alert = int(v["alert"].sum())
    print(f"[pipeline] {len(v):,} sessions -> {n_alert:,} alertes")
    if "label" in df:
        tp = int(((v["alert"] == 1) & (df["label"] == 1)).sum())
        print(f"           dont {tp:,} vrais C2 sur {int((df['label'] == 1).sum()):,} C2 présents")
        for col in [c for c in v.columns if c.endswith("_alert")]:
            print(f"           {col:<14} {int(v[col].sum()):>7,} alertes")
    print("[pipeline] chronométrage (s) :", {k: round(t, 3) for k, t in res["timings"].items()})
    if res["alerts"]:
        print("[pipeline] exemple d'alerte :", res["alerts"][0])


if __name__ == "__main__":
    main()
