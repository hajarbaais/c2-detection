"""P5.7 - Calibration du seuil de l'autoencodeur à 0,1 % de FPR et mesure du recall obtenu.

Pour chaque groupe (dns, web) :
  1. calcule le score (erreur de reconstruction) de TOUTES les sessions avec le modèle entraîné ;
  2. fixe le seuil sur le BÉNIN DE VALIDATION : le plus petit seuil laissant au plus floor(0,1 % x n) faux positifs ;
  3. mesure sur le TEST (jamais utilisé pour choisir) le FPR réalisé et le recall, avec intervalles de Wilson à 95 % ;
  4. indique si le seuil est FIABLE : avec n sessions bénignes en validation, 0,1 % = n/1000 fausses alertes autorisées ;
     en dessous de 10 (n < 10 000), le seuil est instable ;
  5. ajoute un seuil « strict » (règle des trois) quand n >= 3 000 : seuil sans aucun faux positif en validation ;
  6. donne aussi le compromis à 0,5 % et 1 % de FPR, et le recall par famille ;
  7. pour le DNS (option --p3) : recall sans les requêtes DNS banales de CTU-13 (`ctu13_broad_label`).

Le seuil recalculé doit être égal à celui stocké dans le .pt (même données, même procédure) : un écart signale que les
features ont changé depuis l'entraînement (par exemple après les corrections de P5) -> réentraîner.

Sorties (suffixe `_resplit` avec --resplit-benign-web) :
    p6_models/evaluation/ae_calibration.csv            une ligne par groupe, mode et FPR cible
    p6_models/evaluation/ae_calibration_by_family.csv  recall / FPR par famille (test, seuil 0,1 %)
    p6_models/evaluation/ae_thresholds.json            seuils retenus + provenance (petit, versionné)
    p6_models/output/ae_scores.parquet                 score + alerte de chaque session (gitignoré : à partager par Drive)

Usage : python -I p6_models/scripts/p5_7_calibration.py --p2 <dossier sessions_pcap> --p5 <p5_output> [--p3 <sessions_unified.parquet>]
        python -I p6_models/scripts/p5_7_calibration.py ... --resplit-benign-web     (modèle autoencoder_resplit.pt)
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from p5_6a_autoencoder_proto import P5_COLS  # noqa: E402
from p5_6b_autoencoder_train import (load_autoencoder, recompute_p5_on_benign,  # noqa: E402
                                     resplit_benign_web, score_autoencoder)
from p6_common import (EVAL_DIR, MODELS_DIR, REPO, TARGET_FPR, git_commit, load_dataset,  # noqa: E402
                       p2_dir, setup_mlflow)
from metrics import metrics_by, threshold_at_fpr, wilson_ci  # noqa: E402

TARGETS = [0.001, 0.005, 0.01]     # FPR cibles ; 0,1 % est l'objectif du professeur
TARGET_RECALL = 0.98
MIN_FP_FOR_STABLE = 10             # moins de 10 fausses alertes autorisées en validation : seuil instable


# --------------------------------------------------------------------------- fonctions pures (sans PyTorch)
def fp_allowed(n: int, target: float) -> int:
    return int(np.floor(target * n + 1e-9))


def threshold_strict(benign_scores: np.ndarray, target: float):
    """Seuil sans aucun faux positif en validation. Utilisable seulement si n >= 3/target (règle des trois :
    0 faux positif sur n sessions donne FPR < 3/n à 95 %)."""
    benign_scores = np.asarray(benign_scores, dtype=float)
    return float(np.nanmax(benign_scores)) if benign_scores.size >= 3.0 / target else None


def rates(g: pd.DataFrame, thr: float) -> dict:
    """FPR et recall de g (colonnes label, ae_score) avec intervalles de Wilson."""
    y = g["label"].to_numpy()
    pred = g["ae_score"].to_numpy() > thr
    n_b, n_c = int((y == 0).sum()), int((y == 1).sum())
    fp, tp = int((pred & (y == 0)).sum()), int((pred & (y == 1)).sum())
    f_lo, f_hi = wilson_ci(fp, n_b)
    r_lo, r_hi = wilson_ci(tp, n_c)
    return dict(n_benign=n_b, fp=fp, fpr=fp / n_b if n_b else np.nan, fpr_low=f_lo, fpr_high=f_hi,
                n_c2=n_c, tp=tp, recall=tp / n_c if n_c else np.nan, recall_low=r_lo, recall_high=r_hi)


def operating_points(g: pd.DataFrame, group: str) -> list[dict]:
    """Points de fonctionnement d'un groupe : seuils calibrés sur le bénin de validation, mesurés sur val et test."""
    val, test = g[g["split"] == "val"], g[g["split"] == "test"]
    val_b = val.loc[val["label"] == 0, "ae_score"].to_numpy()
    plan = [(t, "quantile", threshold_at_fpr(val_b, t)) for t in TARGETS]
    strict = threshold_strict(val_b, TARGETS[0])
    if strict is not None:
        plan.append((TARGETS[0], "strict_regle_des_trois", strict))
    has_ctu = group == "dns" and "ctu13_broad_label" in g
    rows = []
    for target, mode, thr in plan:
        strict_mode = mode.startswith("strict")      # seuil sans faux positif : fiable par construction (n >= 3 000)
        k = 0 if strict_mode else fp_allowed(len(val_b), target)
        row = dict(group=group, mode=mode, target_fpr=target, n_val_benign=len(val_b), fp_allowed_val=k,
                   threshold_reliable=True if strict_mode else bool(k >= MIN_FP_FOR_STABLE), threshold=thr)
        for name, part in (("val", val), ("test", test)):
            row.update({f"{name}_{k2}": v for k2, v in rates(part, thr).items()})
        if has_ctu:
            keep = test[~test["ctu13_broad_label"].fillna(False).astype(bool)]
            row.update({f"test_sans_ctu13_{k2}": v for k2, v in rates(keep, thr).items()})
        row["meets_fpr_target"] = bool(row["test_fpr"] < TARGET_FPR)
        row["meets_recall_target"] = bool(row["test_recall"] >= TARGET_RECALL)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- données
def build_dataframe(a) -> pd.DataFrame:
    """Même préparation que p5_6b_autoencoder_train.py : le modèle attend exactement ces colonnes."""
    df = load_dataset(p2_dir(a.p2)).reset_index(drop=True)
    if a.resplit_benign_web:
        df = resplit_benign_web(df)
    if a.p5 and "p5_ja3_rarity" not in df.columns:
        t = pd.read_parquet(Path(a.p5) / "features_tls.parquet")[["session_id"] + P5_COLS]
        df = df.merge(t.rename(columns={c: "p5_" + c for c in P5_COLS}), on="session_id", how="left",
                      validate="one_to_one")
        if not a.p5_global_rarity:
            df = recompute_p5_on_benign(df)
    if a.p3:
        u = pd.read_parquet(a.p3, columns=["session_id", "ctu13_broad_label"])
        df = df.merge(u, on="session_id", how="left", validate="one_to_one")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2", help="dossier contenant sessions_pcap.parquet")
    ap.add_argument("--p5", help="dossier p5_output (features_tls.parquet)")
    ap.add_argument("--p3", help="sessions_unified.parquet (ctu13_broad_label, pour l'évaluation DNS)")
    ap.add_argument("--model", help="fichier .pt (défaut : autoencoder.pt, ou autoencoder_resplit.pt)")
    ap.add_argument("--resplit-benign-web", action="store_true",
                    help="même redécoupage expérimental que l'entraînement (utiliser le modèle _resplit)")
    ap.add_argument("--p5-global-rarity", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()

    suffix = "_resplit" if a.resplit_benign_web else ""
    model_path = Path(a.model) if a.model else MODELS_DIR / f"autoencoder{suffix}.pt"
    print(f"Modèle : {model_path}")
    if a.resplit_benign_web != ("resplit" in model_path.name):
        print("  ! le découpage (--resplit-benign-web) et le modèle ne correspondent pas : calibration non valide.")

    df = build_dataframe(a)
    loaded = load_autoencoder(model_path)
    df["ae_score"] = score_autoencoder(loaded, df)["ae_score"].to_numpy()

    rows, fams, thresholds = [], [], {}
    for group in loaded:
        g = df[df["group"] == group]
        ops = operating_points(g, group)
        rows += ops
        main_op = next(r for r in ops if r["mode"] == "quantile" and r["target_fpr"] == TARGET_FPR)
        thr = main_op["threshold"]
        gap = abs(thr - loaded[group]["threshold"])
        print(f"\n=== {group} : {main_op['n_val_benign']:,} bénins en validation -> "
              f"{main_op['fp_allowed_val']} fausses alertes autorisées "
              f"({'seuil fiable' if main_op['threshold_reliable'] else 'SEUIL PEU FIABLE'})")
        print(f"    seuil recalculé {thr:.6f} ; seuil du .pt {loaded[group]['threshold']:.6f}"
              + ("   ! écart : les features ont changé depuis l'entraînement, réentraîner" if gap > 1e-4 * max(1, thr) else ""))
        test = g[g["split"] == "test"]
        fams.append(metrics_by(test.assign(pred=test["ae_score"] > thr), "family").assign(group=group))
        thresholds[group] = {k: main_op[k] for k in ("threshold", "n_val_benign", "fp_allowed_val", "threshold_reliable",
                                                       "val_fpr", "test_fpr", "test_fpr_low", "test_fpr_high",
                                                       "test_recall", "test_recall_low", "test_recall_high",
                                                       "meets_fpr_target", "meets_recall_target")}
        df.loc[g.index, "ae_threshold"] = thr

    res = pd.DataFrame(rows)
    df["ae_alert"] = df["ae_score"] > df["ae_threshold"]
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = REPO / "p6_models" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(EVAL_DIR / f"ae_calibration{suffix}.csv", index=False, float_format="%.6f")
    pd.concat(fams).to_csv(EVAL_DIR / f"ae_calibration_by_family{suffix}.csv", index=False, float_format="%.6f")
    meta = {"model": model_path.name, "split_mode": "resplit_benign_web (expérience)" if suffix else "split P2",
            "date": date.today().isoformat(), "git_commit": git_commit(), "target_fpr": TARGET_FPR,
            "seuil_regle": "bénin de validation, plus petit seuil laissant <= floor(0,1 % x n) faux positifs",
            "groups": thresholds}
    (EVAL_DIR / f"ae_thresholds{suffix}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    keep = ["session_id", "group", "split", "label", "family", "ae_score", "ae_threshold", "ae_alert"]
    keep += [c for c in ("unseen_strict", "ctu13_broad_label") if c in df]
    df[keep].to_parquet(out_dir / f"ae_scores{suffix}.parquet", index=False)

    if not a.no_mlflow and (mlflow := setup_mlflow("c2-autoencoder")):
        with mlflow.start_run(run_name=f"calibration{suffix}"):
            mlflow.set_tags({"lot": "P5", "task": "P5.7", "author": "oumaima", "git_commit": git_commit()})
            for r in rows:
                if r["mode"] == "quantile":
                    tag = f"{r['group']}/fpr{r['target_fpr']}"
                    for k in ("threshold", "test_fpr", "test_recall", "val_fpr"):
                        mlflow.log_metric(f"calib/{tag}/{k}", float(r[k]))
            mlflow.log_artifact(str(EVAL_DIR / f"ae_calibration{suffix}.csv"))

    pd.set_option("display.width", 220)
    view = res[["group", "mode", "target_fpr", "n_val_benign", "fp_allowed_val", "threshold_reliable", "test_n_benign",
                "test_fpr", "test_fpr_low", "test_fpr_high", "test_recall", "test_recall_low", "test_recall_high"]]
    print("\n=== Résultats (seuil calibré sur le bénin de validation, mesure sur le test)")
    print(view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if "test_sans_ctu13_recall" in res:
        print("\n=== DNS sans les requêtes DNS banales de CTU-13")
        print(res[res.group == "dns"][["mode", "target_fpr", "test_sans_ctu13_n_c2", "test_sans_ctu13_tp",
                                       "test_sans_ctu13_recall", "test_sans_ctu13_recall_low"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nObjectifs du prof : recall >= {TARGET_RECALL} et FPR < {TARGET_FPR} (mesurés sur le test, jamais annoncés avant).")
    print(f"Fichiers : {EVAL_DIR / ('ae_calibration' + suffix + '.csv')}, ae_thresholds{suffix}.json, "
          f"{out_dir / ('ae_scores' + suffix + '.parquet')}")


if __name__ == "__main__":
    main()
