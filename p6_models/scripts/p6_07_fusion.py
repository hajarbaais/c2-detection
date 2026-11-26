"""P6.6 - Fusion des scores : autoencodeur (P5) + modèle(s) séquentiel(s) (LSTM P6.3, Transformer P6.5).

Autonome : ne lit que des fichiers parquet de scores (pandas, numpy, pyarrow ; joblib pour la sauvegarde).

Principe
  * Chaque détecteur est converti en RANG par rapport au bénin de validation de sa population (ECDF) : u dans [0, 1],
    proche de 1 = plus extrême que presque tout le bénin de validation. Les échelles deviennent comparables.
  * Règles testées : max (alerte si l'un OU l'autre est extrême), mean, and (les deux doivent l'être), fisher
    (somme de -log(1-u)).
  * Un seul seuil par population, choisi sur le BÉNIN DE VALIDATION pour un FPR cible (0,1 %). Le test n'intervient
    jamais dans un choix.
  * Populations (strates), chacune calibrée séparément :
      dns        : autoencodeur seul (aucun modèle séquentiel utile pour le DNS)
      web_seq    : web avec séquence valide -> fusion autoencodeur + séquentiel(s)
      web_noseq  : web sans séquence valide (sessions « fallback ») -> autoencodeur seul
  * La combinaison (détecteurs x règle) est choisie sur le RECALL DE VALIDATION de la strate web_seq au FPR cible.
    Tous les candidats sont toutefois rapportés sur le test, pour la transparence.

Entrées
  --ae   ae_scores.parquet (P5.7)                      colonnes : session_id, group, split, label, family, ae_score ...
  --seq  nom=chemin.parquet:colonne    (répétable)     ex. transformer=transformer_scores_packets.parquet:transformer_score
                                                            lstm=lstm_scores.parquet:lstm_score
         Un score est ignoré (vide) si le fichier contient fallback_only = True ou si group != web.

Sorties
  p6_models/evaluation/fusion_metrics.csv    tous les candidats : recall / FPR avec IC, par strate et agrégé
  p6_models/evaluation/fusion_by_family.csv  recall / FPR par famille (configuration retenue, test)
  p6_models/evaluation/fusion_rule.json      configuration retenue + provenance (petit, versionné)
  p6_models/output/scores.parquet            score fusionné, alerte et verdict de chaque session (gitignoré)
  p6_models/models/fusion_reference.joblib   seuils + références ECDF pour le pipeline (gitignoré) : apply_fusion()

Usage :
  python -I p6_models/scripts/p6_07_fusion.py --ae p6_models/output/ae_scores.parquet \\
      --seq transformer=<...>/transformer_scores_packets.parquet:transformer_score \\
      --seq lstm=<...>/lstm_scores.parquet:lstm_score
"""
import argparse
import itertools
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RULES = ["max", "mean", "and", "fisher"]
TARGET_RECALL = 0.98
MIN_FP_FOR_STABLE = 10


# --------------------------------------------------------------------------- outils statistiques
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def threshold_at_fpr(benign_scores, target: float) -> float:
    """Plus petit seuil laissant au plus floor(target x n) bénins strictement au-dessus (alerte si score > seuil)."""
    s = np.sort(np.asarray(benign_scores, dtype=float))
    s = s[~np.isnan(s)]
    if s.size == 0:
        return math.inf
    k = int(math.floor(target * s.size + 1e-9))
    return float(s[s.size - k - 1])


def ecdf(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rang de x parmi le bénin de validation : (nombre de références <= x) / (n + 1).

    Au-delà du maximum de validation, le rang plafonnerait (plusieurs sessions ex æquo, donc indiscernables) : on le
    prolonge d'un terme strictement croissant, inférieur à la moitié d'un pas de rang, qui ne change pas l'ordre des
    rangs ordinaires.
    """
    n = len(ref_sorted)
    u = np.searchsorted(ref_sorted, x, side="right") / (n + 1.0)
    top, med = ref_sorted[-1], ref_sorted[n // 2]
    scale = max(float(top - med), 1e-12)
    excess = np.clip((x - top) / scale, 0.0, 700.0)
    return u + np.where(x > top, (1.0 - np.exp(-excess)) / (2.0 * (n + 1.0)), 0.0)


def fuse(U: np.ndarray, rule: str, n_ref: int) -> np.ndarray:
    """U : tableau (détecteurs x sessions) de rangs. Score fusionné, grand = suspect."""
    if rule == "max":
        return U.max(axis=0)
    if rule == "mean":
        return U.mean(axis=0)
    if rule == "and":
        return U.min(axis=0)
    if rule == "fisher":
        eps = 0.25 / (n_ref + 1.0)          # plus petit que 1/(2(n+1)) : garde le prolongement de ecdf()
        return (-np.log(np.clip(1.0 - U, eps, 1.0))).sum(axis=0)
    raise ValueError(rule)


def rates(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y)
    pred = np.asarray(pred, dtype=bool)
    n_b, n_c = int((y == 0).sum()), int((y == 1).sum())
    fp, tp = int((pred & (y == 0)).sum()), int((pred & (y == 1)).sum())
    return _rate_dict(fp, n_b, tp, n_c)


def _rate_dict(fp, n_b, tp, n_c) -> dict:
    f_lo, f_hi = wilson_ci(fp, n_b)
    r_lo, r_hi = wilson_ci(tp, n_c)
    return dict(n_benign=n_b, fp=fp, fpr=fp / n_b if n_b else math.nan, fpr_low=f_lo, fpr_high=f_hi,
                n_c2=n_c, tp=tp, recall=tp / n_c if n_c else math.nan, recall_low=r_lo, recall_high=r_hi)


# --------------------------------------------------------------------------- données
def load_ae(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"session_id", "group", "split", "label", "ae_score"}
    if need - set(df.columns):
        raise SystemExit(f"{path} : colonnes manquantes {sorted(need - set(df.columns))}")
    if df["session_id"].duplicated().any():
        raise SystemExit(f"{path} : session_id en double")
    return df.reset_index(drop=True)


def load_seq_scores(spec: str, session_ids: pd.Series):
    """'nom=chemin.parquet:colonne' -> (nom, Series de scores alignée sur session_ids, vide hors web valide)."""
    name, rest = spec.split("=", 1)
    path, col = rest.rsplit(":", 1)
    s = pd.read_parquet(path)
    if col not in s.columns:
        raise SystemExit(f"{path} : colonne '{col}' introuvable (colonnes : {list(s.columns)})")
    if s["session_id"].duplicated().any():
        raise SystemExit(f"{path} : session_id en double")
    s = s.set_index("session_id")
    invalid = pd.Series(False, index=s.index)
    if "fallback_only" in s:
        invalid |= s["fallback_only"].astype(bool)
    if "group" in s:
        invalid |= s["group"] != "web"
    score = s[col].astype(float).mask(invalid)
    out = session_ids.map(score)
    return name, out.to_numpy(dtype=float)


# --------------------------------------------------------------------------- strates et configurations
def strata_masks(df: pd.DataFrame, mask_names: list[str], web_dets: list[str]) -> dict:
    """Populations : la strate web_seq = web dont TOUS les scores séquentiels de mask_names sont valides."""
    dns = (df["group"] == "dns").to_numpy()
    web = (df["group"] == "web").to_numpy()
    seq_ok = np.zeros(len(df), dtype=bool)
    if mask_names:
        seq_ok = web & df[[f"{n}_score" for n in mask_names]].notna().all(axis=1).to_numpy()
    return {"dns": (dns, ["ae"]), "web_seq": (seq_ok, web_dets), "web_noseq": (web & ~seq_ok, ["ae"])}


def run_config(df: pd.DataFrame, mask_names: list[str], web_dets: list[str], rule: str, target: float,
               noseq_policy: str = "ae"):
    """Une configuration (détecteurs du web_seq, règle). Renvoie (lignes de résultats, sorties par session, références)."""
    n = len(df)
    fused = np.full(n, np.nan)
    thr_arr = np.full(n, np.nan)
    alert = np.zeros(n, dtype=bool)
    stratum = np.array([""] * n, dtype=object)
    rows, refs = [], {}
    for name, (m, dets) in strata_masks(df, mask_names, web_dets).items():
        if not m.any():
            continue
        sub = df[m]
        is_val_b = ((sub["split"] == "val") & (sub["label"] == 0)).to_numpy()
        if is_val_b.sum() == 0:
            continue
        if len(dets) == 1:                      # un seul détecteur : le score brut suffit (même seuil que P5.7)
            f = sub[f"{dets[0]}_score"].to_numpy(dtype=float)
            ref = {dets[0]: np.sort(f[is_val_b])}
        else:
            ref = {d: np.sort(sub[f"{d}_score"].to_numpy(dtype=float)[is_val_b]) for d in dets}
            U = np.vstack([ecdf(ref[d], sub[f"{d}_score"].to_numpy(dtype=float)) for d in dets])
            f = fuse(U, rule, int(is_val_b.sum()))
        thr = threshold_at_fpr(f[is_val_b], target)
        abstain = bool(name == "web_noseq" and noseq_policy == "abstain")   # pas d'alerte : sessions non évaluables
        a = (f > thr) & (not abstain)
        fused[m], thr_arr[m], alert[m], stratum[m] = f, thr, a, name
        k = int(math.floor(target * is_val_b.sum() + 1e-9))
        row = dict(stratum=name, detectors="+".join(dets), rule=rule if len(dets) > 1 else "-", cand_rule=rule,
                   target_fpr=target,
                   n_val_benign=int(is_val_b.sum()), fp_allowed_val=k,
                   threshold_reliable=bool(k >= MIN_FP_FOR_STABLE), threshold=thr)
        y, split = sub["label"].to_numpy(), sub["split"].to_numpy()
        for sp in ("val", "test"):
            row.update({f"{sp}_{k2}": v for k2, v in rates(y[split == sp], a[split == sp]).items()})
        un = split == "unseen"
        row["unseen_n_c2"], row["unseen_tp"] = int(un.sum()), int(a[un].sum())
        if "unseen_strict" in sub:
            us = (split == "unseen") & sub["unseen_strict"].fillna(False).astype(bool).to_numpy()
            row["unseen_strict_n_c2"], row["unseen_strict_tp"] = int(us.sum()), int(a[us].sum())
        rows.append(row)
        row["abstain"] = abstain
        refs[name] = {"detectors": dets, "rule": rule if len(dets) > 1 else None, "threshold": float(thr), "abstain": abstain,
                      "n_val_benign": int(is_val_b.sum()), "refs": {d: v for d, v in ref.items()}}
    return rows, {"fused_score": fused, "fused_threshold": thr_arr, "fused_alert": alert, "stratum": stratum}, refs


def pooled(rows: list[dict], strata: list[str], label: str) -> dict:
    sel = [r for r in rows if r["stratum"] in strata]
    if not sel:
        return {}
    out = dict(stratum=label, detectors="", rule=sel[0]["rule"], cand_rule=sel[0]["cand_rule"],
               target_fpr=sel[0]["target_fpr"],
               n_val_benign=sum(r["n_val_benign"] for r in sel),
               fp_allowed_val=sum(r["fp_allowed_val"] for r in sel),
               threshold_reliable=all(r["threshold_reliable"] for r in sel), threshold=math.nan)
    for sp in ("val", "test"):
        out.update({f"{sp}_{k}": v for k, v in _rate_dict(
            sum(r[f"{sp}_fp"] for r in sel), sum(r[f"{sp}_n_benign"] for r in sel),
            sum(r[f"{sp}_tp"] for r in sel), sum(r[f"{sp}_n_c2"] for r in sel)).items()})
    out["unseen_n_c2"] = sum(r["unseen_n_c2"] for r in sel)
    out["unseen_tp"] = sum(r["unseen_tp"] for r in sel)
    if all("unseen_strict_n_c2" in r for r in sel):
        out["unseen_strict_n_c2"] = sum(r["unseen_strict_n_c2"] for r in sel)
        out["unseen_strict_tp"] = sum(r["unseen_strict_tp"] for r in sel)
    return out


# --------------------------------------------------------------------------- application (pipeline P7)
def apply_fusion(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Applique une configuration retenue (fusion_reference.joblib) à des sessions.

    df : colonnes `group`, `ae_score` et `<nom>_score` pour chaque séquentiel utilisé (vide si pas de séquence).
    Renvoie fused_score, fused_alert, stratum, verdict, dans l'ordre de df.
    """
    seq_names = [d for d in ref["web_seq"]["detectors"] if d != "ae"] if "web_seq" in ref else []
    # la population web_seq est définie par les scores séquentiels valides (voir mask_names au calibrage)
    if "web_seq" in ref and "mask_names" in ref["web_seq"]:
        seq_names = list(ref["web_seq"]["mask_names"])
    n = len(df)
    out = pd.DataFrame({"fused_score": np.nan, "fused_alert": False, "stratum": ""}, index=df.index)
    web = (df["group"] == "web").to_numpy()
    seq_ok = np.zeros(n, dtype=bool)
    if seq_names:
        seq_ok = web & df[[f"{s}_score" for s in seq_names]].notna().all(axis=1).to_numpy()
    for name, m in (("dns", (df["group"] == "dns").to_numpy()), ("web_seq", seq_ok), ("web_noseq", web & ~seq_ok)):
        if name not in ref or not m.any():
            continue
        r = ref[name]
        sub = df[m]
        if len(r["detectors"]) == 1:
            f = sub[f"{r['detectors'][0]}_score"].to_numpy(dtype=float)
        else:
            U = np.vstack([ecdf(r["refs"][d], sub[f"{d}_score"].to_numpy(dtype=float)) for d in r["detectors"]])
            f = fuse(U, r["rule"], r["n_val_benign"])
        out.loc[m, "fused_score"] = f
        out.loc[m, "fused_alert"] = (f > r["threshold"]) & (not r.get("abstain", False))
        out.loc[m, "stratum"] = name
    out["verdict"] = np.where(out["fused_alert"], "ALERTE", "NORMAL")
    abst = [k for k in ("web_noseq",) if k in ref and ref[k].get("abstain", False)]
    if abst:
        out.loc[out["stratum"].isin(abst), "verdict"] = "NON_EVALUE"
    return out


# --------------------------------------------------------------------------- programme principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ae", required=True)
    ap.add_argument("--seq", action="append", default=[], help="nom=chemin.parquet:colonne (répétable)")
    ap.add_argument("--target", type=float, default=0.001)
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--noseq", choices=["ae", "abstain"], default="ae",
                    help="sessions web sans séquence : 'ae' = autoencodeur seul (défaut) ; 'abstain' = aucune alerte "
                         "(sessions non évaluables, comptées comme manquées dans le recall)")
    ap.add_argument("--select", help="forcer la configuration, ex. 'ae+transformer|and' ou 'lstm|-' pour un détecteur seul (défaut : meilleur recall de validation)")
    ap.add_argument("--eval-dir", default=str(REPO / "p6_models" / "evaluation"))
    ap.add_argument("--output-dir", default=str(REPO / "p6_models" / "output"))
    ap.add_argument("--models-dir", default=str(REPO / "p6_models" / "models"))
    a = ap.parse_args()

    df = load_ae(a.ae)
    seq_names = []
    for spec in a.seq:
        name, arr = load_seq_scores(spec, df["session_id"])
        df[f"{name}_score"] = arr
        seq_names.append(name)
        print(f"Séquentiel '{name}' : {int(np.isfinite(arr).sum()):,} sessions avec score valide")

    # candidats : (1) fusions = chaque séquentiel avec l'autoencodeur, puis tous ensemble, x les règles ;
    #             (2) détecteurs SEULS (autoencodeur, chaque séquentiel) sur la même population : référence de comparaison
    rules = [r for r in a.rules.split(",") if r in RULES]
    fusion_sets = [[s] for s in seq_names] + ([seq_names] if len(seq_names) > 1 else [])
    candidates = [(sq, ["ae"] + sq, "ae" + "".join("+" + x for x in sq), r) for sq in fusion_sets for r in rules]
    candidates += [(seq_names, [d], d, "-") for d in ["ae"] + seq_names]
    if not seq_names:
        candidates = [([], ["ae"], "ae", "-")]

    all_rows, results = [], []
    for mask_names, web_dets, tag, rule in candidates:
        rows, per_session, refs = run_config(df, mask_names, web_dets, rule if rule != "-" else "max", a.target,
                                             a.noseq)
        for r in rows:
            r["config"] = tag
            r["cand_rule"] = rule
        if "web_seq" in refs:
            refs["web_seq"]["mask_names"] = list(mask_names)
        pooled_rows = [pooled(rows, ["dns", "web_seq", "web_noseq"], "ALL"), pooled(rows, ["web_seq", "web_noseq"], "WEB"),
                       pooled(rows, ["dns"], "DNS")]
        for r in pooled_rows:
            if r:
                r["config"] = tag
                r["cand_rule"] = rule
        all_rows += rows + [r for r in pooled_rows if r]
        ws = next((r for r in rows if r["stratum"] == "web_seq"), None)
        results.append({"seqs": mask_names, "rule": rule, "tag": tag, "rows": rows, "per_session": per_session,
                        "refs": refs, "val_recall": ws["val_recall"] if ws else -1.0})

    if a.select:
        wanted = [r for r in results if f"{r['tag']}|{r['rule']}" == a.select]
        if not wanted:
            raise SystemExit(f"--select {a.select} inconnu ; candidats : {[f'{r[chr(116)+chr(97)+chr(103)]}|{r[chr(114)+chr(117)+chr(108)+chr(101)]}' for r in results]}")
        best = wanted[0]
    else:
        best = max(results, key=lambda r: r["val_recall"])      # choix sur la VALIDATION uniquement
    res = pd.DataFrame(all_rows)
    first = ["config", "rule", "stratum", "detectors"]
    res = res[first + [c for c in res.columns if c not in first]]
    eval_dir, out_dir, models_dir = Path(a.eval_dir), Path(a.output_dir), Path(a.models_dir)
    for d in (eval_dir, out_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)
    res["retained"] = (res["config"] == best["tag"]) & (res["cand_rule"] == best["rule"])
    res.to_csv(eval_dir / "fusion_metrics.csv", index=False, float_format="%.6f")

    # sorties par session de la configuration retenue
    ps = best["per_session"]
    cols = [c for c in ("session_id", "group", "split", "label", "family", "unseen_strict", "ctu13_broad_label", "ae_score")
            if c in df.columns] + [f"{s}_score" for s in seq_names]
    scores = df[cols].copy()
    for k, v in ps.items():
        scores[k] = v
    scores["verdict"] = np.where(scores["fused_alert"], "ALERTE", "NORMAL")
    if a.noseq == "abstain":
        scores.loc[scores["stratum"] == "web_noseq", "verdict"] = "NON_EVALUE"
    scores.to_parquet(out_dir / "scores.parquet", index=False)

    # par famille (test) pour la configuration retenue
    t = scores[scores["split"] == "test"]
    fam = []
    for (g, f), x in t.groupby(["group", "family"]):
        r = rates(x["label"].to_numpy(), x["fused_alert"].to_numpy())
        fam.append({"group": g, "family": f, **{k: r[k] for k in ("n_c2", "n_benign", "tp", "fp", "recall", "fpr")}})
    pd.DataFrame(fam).to_csv(eval_dir / "fusion_by_family.csv", index=False, float_format="%.6f")

    # configuration retenue + références pour le pipeline
    try:
        import joblib
        joblib.dump(best["refs"], models_dir / "fusion_reference.joblib")
    except ImportError:
        print("joblib absent : fusion_reference.joblib non écrit")
    meta = {"date": date.today().isoformat(), "target_fpr": a.target,
            "choix": ("forcé par --select : " + a.select) if a.select else
                     "combinaison (détecteurs, règle) retenue sur le recall de VALIDATION de la strate web_seq",
            "configuration": best["tag"], "politique_web_sans_sequence": a.noseq, "regle_web_seq": best["rule"] if seq_names else None,
            "strates": {k: {kk: vv for kk, vv in v.items() if kk != "refs"} for k, v in best["refs"].items()},
            "candidats_val_recall_web_seq": {f"{r['tag']}|{r['rule']}": round(r["val_recall"], 4) for r in results}}
    (eval_dir / "fusion_rule.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # affichage
    pd.set_option("display.width", 230)
    show = res[res["stratum"].isin(["web_seq"])][["config", "rule", "n_val_benign", "fp_allowed_val", "threshold_reliable",
                                                   "val_recall", "test_fpr", "test_recall", "test_n_benign", "test_n_c2"]]
    print("\n=== Candidats (strate web_seq ; choix sur val_recall, test pour information)")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n>>> Configuration retenue (validation) : {best['tag']} avec la règle '{best['rule']}'")
    fin = res[res["retained"]]
    cols2 = ["stratum", "n_val_benign", "fp_allowed_val", "threshold_reliable", "test_n_benign", "test_fp", "test_fpr",
             "test_fpr_low", "test_fpr_high", "test_n_c2", "test_tp", "test_recall", "test_recall_low", "test_recall_high"]
    print(fin[cols2].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nObjectifs du prof : recall >= {TARGET_RECALL} et FPR < {a.target} (mesurés sur le test, jamais annoncés avant).")
    print(f"Fichiers : {eval_dir / 'fusion_metrics.csv'}, fusion_by_family.csv, fusion_rule.json ; {out_dir / 'scores.parquet'}")


if __name__ == "__main__":
    main()
