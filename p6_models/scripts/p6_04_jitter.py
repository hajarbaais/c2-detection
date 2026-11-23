"""P6.4 - Robustesse au jitter : bruit réaliste sur les intervalles + évaluation de n'importe quel modèle séquentiel.

Auteure : Wassima (P6.3-P6.4, sprint 3).

Pourquoi ce module ?
  Un C2 « avec jitter » (Cobalt Strike, Sliver…) ne balise pas toutes les 60 s pile mais toutes les
  60 s ± X %. Pour simuler ça il faut modifier l'intervalle EN SECONDES, pas la valeur standardisée :
  P5 stocke log_dt = log1p(dt) puis le standardise avec seq_scaler.pkl. Ajouter un bruit gaussien
  directement sur la valeur standardisée (option --jitter-std du Transformer) ne correspond à aucun
  pourcentage de jitter précis. Ici :

      dt (s) = expm1(x * scale + mean)      # on revient en secondes (scaler P5 de log_dt)
      dt'    = dt * facteur                 # facteur ~ U(1 - j, 1 + j)   (kind = "sym", défaut)
                                            #         ~ U(1 - j, 1)       (kind = "down", style Cobalt Strike)
      x'     = (log1p(dt') - mean) / scale  # et on re-standardise

  j = 0,5 veut dire « jitter 50 % » (objectif P6.7 de Hajar). Les pas de padding, le premier pas
  (dt = 0) et les 5 autres features ne bougent pas.

Deux usages :
  1. Entraînement robuste (P6.4) : p6_03_lstm.py --jitter-train 0.5 tire un niveau ~ U(0, 0,5) par session.
  2. Évaluation de robustesse (ce script, CLI) : pour un modèle déjà entraîné (LSTM ou Transformer),
     on bruite UNIQUEMENT les sessions C2 (c'est l'attaquant qui choisit son jitter), le bénin et le
     seuil (fixé sur le bénin de validation propre) ne changent pas -> la FPR ne bouge pas, on mesure
     la chute du rappel.

Limite connue : les séquences web de P5 mélangent intervalles ENTRE connexions (le vrai battement du
C2) et intervalles entre paquets d'une même connexion. On bruite tous les dt : c'est un test plus
dur que la réalité pour les pas intra-connexion.

Usage (CLI) :
  python -I p6_models/scripts/p6_04_jitter.py --model p6_models/models/lstm_anomaly_packets.pt \
         --p5 p5_features_tls_seq/output --p2 <dossier output P2>
  python -I p6_models/scripts/p6_04_jitter.py --model p6_models/models/tf_web_anomaly_packets.pt \
         --p5 p5_features_tls_seq/output --p2 <dossier output P2>          # Transformer d'Imane
Sorties : p6_models/evaluation/<arch>_<tag>_jitter.csv  (+ _jitter_by_family.csv avec --p2)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from p6_common import EVAL_DIR, RANDOM_STATE, REPO  # noqa: E402
from metrics import binary_metrics, metrics_by  # noqa: E402

DEFAULT_P5 = REPO / "p5_features_tls_seq" / "output"
DT_IDX = 0                                  # log_dt = SEQ_FEATS[0] dans P5
DEFAULT_LEVELS = (0.0, 0.1, 0.25, 0.5)
JITTER_KINDS = ("sym", "down")


# --------------------------------------------------------------------------- jitter
def load_dt_scaler(p5_dir) -> tuple[float, float]:
    """(moyenne, écart-type) de log1p(dt) appris par P5 sur le train (seq_scaler.pkl)."""
    import joblib
    d = joblib.load(Path(p5_dir) / "seq_scaler.pkl")
    sc, feats = d["scaler"], list(d.get("features", ["log_dt", "log_size", "log_resp_size"]))
    i = feats.index("log_dt")
    return float(sc.mean_[i]), float(sc.scale_[i])


def jitter_log_dt(x: torch.Tensor, mask: torch.Tensor, level, dt_mean: float, dt_scale: float,
                  kind: str = "sym", generator: torch.Generator | None = None) -> torch.Tensor:
    """Copie de x (N, L, F) au format P5 où l'intervalle dt de chaque pas valide est multiplié par un facteur aléatoire.

    level : float (même niveau pour toutes les sessions) ou tenseur (N,) (un niveau par session), dans [0, 1].
    kind  : "sym" -> facteur ~ U(1 - level, 1 + level) ; "down" -> U(1 - level, 1) (le sleep ne fait que raccourcir).
    """
    if kind not in JITTER_KINDS:
        raise ValueError(f"kind doit être dans {JITTER_KINDS}")
    n, L = x.shape[0], x.shape[1]
    lv = level if torch.is_tensor(level) else torch.full((n,), float(level))
    lv = lv.detach().to("cpu", torch.float32).clamp(0.0, 1.0).view(n, 1)
    out = x.clone()
    if n == 0 or not bool((lv > 0).any()):
        return out
    u = torch.rand((n, L), generator=generator)                       # tiré sur CPU : reproductible
    lo = 1.0 - lv
    hi = 1.0 + lv if kind == "sym" else torch.ones_like(lv)
    factor = (lo + (hi - lo) * u).to(x.device, x.dtype)

    xd = x[:, :, DT_IDX]
    dt = torch.expm1(xd * dt_scale + dt_mean).clamp(min=0.0)
    new = (torch.log1p(dt * factor) - dt_mean) / dt_scale
    keep = (~mask.bool()) | (lv.to(x.device) == 0)                     # padding et sessions sans jitter : intacts
    out[:, :, DT_IDX] = torch.where(keep, xd, new)
    return out


# --------------------------------------------------------------------------- sélection (identique à P6.5)
def select_group(X: np.ndarray, mask: np.ndarray, meta: pd.DataFrame, group: str, subset: str, max_len: int):
    """Mêmes sessions que p6_05_transformer.run_group : groupe, sous-ensemble `packets` (web), troncature.

    -> (Xg tensor, Mg tensor, g DataFrame) ; Mg passe par fix_empty (1 pas pour les sessions vides).
    """
    from p6_05_transformer import fix_empty
    sel = meta["group"].eq(group).to_numpy()
    if group == "web" and subset == "packets":
        sel = sel & (~meta["fallback_only"] & ~meta["empty"]).to_numpy()
    g = meta[sel].reset_index(drop=True)
    L = min(max_len, X.shape[1])
    Xg = torch.from_numpy(np.ascontiguousarray(X[sel][:, :L]))
    Mg = torch.from_numpy(fix_empty(mask[sel][:, :L]))
    return Xg, Mg, g


# --------------------------------------------------------------------------- évaluation
def robustness(model, Xg, Mg, g: pd.DataFrame, thr: float, levels, dt_mean: float, dt_scale: float,
               kind: str = "sym", seed: int = RANDOM_STATE, device="cpu", group: str = "",
               clean_scores: np.ndarray | None = None):
    """Rappel sous jitter, à seuil FIXE. Bénin = scores propres ; C2 = scores après jitter.

    -> (lignes de métriques par split et niveau, DataFrame par famille sur le test ou None)
    """
    from p6_05_transformer import predict           # marche pour le LSTM et le Transformer (même interface)
    sp, lab = g["split"].to_numpy(), g["label"].to_numpy()
    if clean_scores is None:
        clean_scores = predict(model, Xg, Mg, device)
    splits = [("test", sp == "test"), ("unseen", sp == "unseen")]
    if "unseen_strict" in g:
        splits.append(("unseen_strict", (sp == "unseen") & g["unseen_strict"].fillna(False).astype(bool).to_numpy()))

    rows, fams = [], []
    for split, m in splits:
        if not m.any():
            continue
        ben, c2 = np.flatnonzero(m & (lab == 0)), np.flatnonzero(m & (lab == 1))
        for level in levels:
            gen = torch.Generator().manual_seed(seed)
            if len(c2):
                Xj = jitter_log_dt(Xg[c2], Mg[c2], float(level), dt_mean, dt_scale, kind, gen)
                s_c2 = predict(model, Xj, Mg[c2], device)
            else:
                s_c2 = np.zeros(0)
            idx = np.r_[ben, c2]
            s = np.r_[clean_scores[ben], s_c2]
            y = lab[idx]
            rows.append({"group": group, "split": split, "jitter": float(level), "jitter_kind": kind,
                         "threshold": float(thr), **binary_metrics(y, s > thr, s)})
            if split == "test" and "family" in g:
                fams.append(metrics_by(g.iloc[idx].assign(pred=s > thr), "family")
                            .assign(group=group, jitter=float(level), jitter_kind=kind))
    return rows, (pd.concat(fams, ignore_index=True) if fams else None)


def merge_csv(path: Path, new: pd.DataFrame, key: str = "group") -> pd.DataFrame:
    """Écrit `new` dans `path` en remplaçant seulement les lignes des groupes présents dans `new`.

    Évite qu'un essai `--groups dns` écrase les résultats web du même tag (limite notée dans README_transformer).
    """
    if path.exists() and key in new:
        old = pd.read_csv(path)
        if key in old:
            new = pd.concat([old[~old[key].isin(new[key].unique())], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, float_format="%.6f")
    return new


def load_any(path, device="cpu"):
    """Recharge un .pt LSTM (p6_03) ou Transformer (p6_05) -> (arch, checkpoint, {groupe: {model, threshold, mode}})."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "transformer")           # les .pt d'Imane n'ont pas de clé "arch"
    if arch == "lstm":
        from p6_03_lstm import load_lstm
        return arch, ckpt, load_lstm(path, device)
    from p6_05_transformer import load_transformer
    return arch, ckpt, load_transformer(path, device)


def ckpt_tag(arch: str, ckpt: dict) -> str:
    tag = f"{arch}_{ckpt.get('mode', 'na')}_{ckpt.get('subset', 'all')}"
    jt = float(ckpt.get("jitter_train", 0) or 0)
    return tag + (f"_jit{round(jt * 100)}" if jt > 0 else "")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=".pt du LSTM (p6_03) ou du Transformer (p6_05)")
    ap.add_argument("--p5", default=str(DEFAULT_P5), help="dossier contenant features_seq.npz et seq_scaler.pkl")
    ap.add_argument("--p2", help="dossier P2 (sessions_pcap.parquet) : rappel par famille + unseen_strict")
    ap.add_argument("--levels", default=",".join(str(v) for v in DEFAULT_LEVELS), help="niveaux de jitter (0,5 = 50 %%)")
    ap.add_argument("--kind", choices=JITTER_KINDS, default="sym")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    a = ap.parse_args()

    from p6_05_transformer import load_seq
    device = "cpu"
    levels = [float(v) for v in a.levels.split(",") if v.strip()]
    arch, ckpt, loaded = load_any(a.model, device)
    X, mask, meta, _ = load_seq(Path(a.p5), Path(a.p2) if a.p2 else None)
    dt_mean, dt_scale = load_dt_scaler(a.p5)
    subset = ckpt.get("subset", "all")
    tag = ckpt_tag(arch, ckpt)
    print(f"{arch} ({tag}) ; log_dt : moyenne {dt_mean:.3f}, écart-type {dt_scale:.3f} ; niveaux {levels} ({a.kind})")

    rows, fams = [], []
    for group, b in loaded.items():
        Xg, Mg, g = select_group(X, mask, meta, group, subset, b["model"].seq_len)
        r, f = robustness(b["model"], Xg, Mg, g, b["threshold"], levels, dt_mean, dt_scale, a.kind, a.seed,
                          device, group)
        rows += r
        if f is not None:
            fams.append(f)
    res = merge_csv(EVAL_DIR / f"{tag}_jitter.csv", pd.DataFrame(rows))
    if fams:
        merge_csv(EVAL_DIR / f"{tag}_jitter_by_family.csv", pd.concat(fams, ignore_index=True))
    view = res[["group", "split", "jitter", "n_c2", "recall", "fpr", "roc_auc"]]
    print(view.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {EVAL_DIR / f'{tag}_jitter.csv'}")


if __name__ == "__main__":
    main()
