"""Tableau comparatif LSTM (P6.3-P6.4, Wassima) vs Transformer (P6.5, Imane) : détection, robustesse, vitesse, taille.

Ne réentraîne rien : lit les CSV déjà écrits dans p6_models/evaluation/ par
  p6_03_lstm.py        lstm_<tag>_{metrics,benchmark,jitter}.csv
  p6_05_transformer.py transformer_<tag>_{metrics,benchmark}.csv
  p6_04_jitter.py      transformer_<tag>_jitter.csv   (robustesse du Transformer, même protocole que le LSTM)
et aligne les essais qui ont le même <tag> (= <mode>_<subset>) et le même groupe (dns / web).
Les variantes entraînées avec jitter (lstm_<tag>_jitNN) sont mises à côté de leur essai de base.

Toutes les valeurs viennent du TEST (seuil fixé sur le bénin de validation), unseen = familles jamais vues.
Une case « – » veut dire : CSV absent (essai non lancé ou non versionné), pas « 0 ».

Usage : python -I p6_models/scripts/p6_compare_seq.py
Sorties : p6_models/evaluation/comparison_lstm_transformer.csv  (format long)
          p6_models/comparison_lstm_transformer.md             (tableaux prêts pour le rapport)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_common import EVAL_DIR, REPO  # noqa: E402

ARCHS = ("lstm", "transformer")
JIT_MAX = 0.5                                   # « jitter 50 % » du cahier des charges (P6.7)
MD_OUT = REPO / "p6_models" / "comparison_lstm_transformer.md"
CSV_OUT = EVAL_DIR / "comparison_lstm_transformer.csv"


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _get(df, split, col, group, **eq):
    if df is None or col not in df:
        return np.nan
    m = (df["group"] == group) & (df["split"] == split)
    for k, v in eq.items():
        m &= np.isclose(df[k], v) if isinstance(v, float) else (df[k] == v)
    return float(df.loc[m, col].iloc[0]) if m.any() else np.nan


def collect(eval_dir: Path | None = None) -> pd.DataFrame:
    """Une ligne par (arch, tag, groupe) avec toutes les colonnes du tableau comparatif."""
    eval_dir = Path(eval_dir or EVAL_DIR)
    rows = []
    for f in sorted(eval_dir.glob("*_metrics.csv")):
        m = re.match(rf"^({'|'.join(ARCHS)})_(.+)_metrics\.csv$", f.name)
        if not m:
            continue
        arch, tag = m.groups()
        met = pd.read_csv(f)
        bench = _read(eval_dir / f"{arch}_{tag}_benchmark.csv")
        jit = _read(eval_dir / f"{arch}_{tag}_jitter.csv")
        base = re.sub(r"_jit\d+$", "", tag)
        for group in sorted(met["group"].unique()):
            r = {"arch": arch, "tag": tag, "base_tag": base, "group": group,
                 "jitter_train": int(re.search(r"_jit(\d+)$", tag).group(1)) / 100 if re.search(r"_jit\d+$", tag) else 0.0}
            for k in ("n_c2", "recall", "fpr", "roc_auc", "pr_auc", "recall_ci_low", "recall_ci_high"):
                r[f"test_{k}"] = _get(met, "test", k, group)
            r["unseen_recall"] = _get(met, "unseen", "recall", group)
            r["unseen_strict_recall"] = _get(met, "unseen_strict", "recall", group)
            r["test_recall_jit0"] = _get(jit, "test", "recall", group, jitter=0.0)
            r[f"test_recall_jit{round(JIT_MAX * 100)}"] = _get(jit, "test", "recall", group, jitter=JIT_MAX)
            if bench is not None and group in set(bench["group"]):
                b = bench[bench["group"] == group].iloc[0]
                for k in ("n_params", "size_mb", "ms_per_session_batch1", "ms_per_session_batch256", "fit_seconds"):
                    r[k] = float(b[k]) if k in b and pd.notna(b[k]) else np.nan
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        j = f"test_recall_jit{round(JIT_MAX * 100)}"
        out["jitter_recall_drop"] = out["test_recall_jit0"] - out[j]
    return out


# --------------------------------------------------------------------------- markdown
def _pct(v, d=1):
    return "–" if pd.isna(v) else f"{100 * v:.{d}f} %".replace(".", ",")


def _num(v, fmt="{:,.0f}"):
    return "–" if pd.isna(v) else fmt.format(v).replace(",", " ").replace(".", ",")


LINES = [  # (libellé, colonne, formateur)
    ("Rappel test", "test_recall", _pct),
    ("IC 95 % du rappel", None, None),
    ("FPR test", "test_fpr", lambda v: _pct(v, 3)),
    ("ROC-AUC test", "test_roc_auc", lambda v: _num(v, "{:.3f}")),
    ("PR-AUC test", "test_pr_auc", lambda v: _num(v, "{:.3f}")),
    ("Rappel unseen", "unseen_recall", _pct),
    (f"Rappel test, C2 avec jitter {round(JIT_MAX * 100)} %", f"test_recall_jit{round(JIT_MAX * 100)}", _pct),
    ("Perte de rappel due au jitter (points)", "jitter_recall_drop", lambda v: _num(100 * v, "{:.1f}") if pd.notna(v) else "–"),
    ("Paramètres", "n_params", _num),
    ("Taille (Mo)", "size_mb", lambda v: _num(v, "{:.2f}")),
    ("Latence CPU, lot de 1 (ms/session)", "ms_per_session_batch1", lambda v: _num(v, "{:.2f}")),
    ("Latence CPU, lot de 256 (ms/session)", "ms_per_session_batch256", lambda v: _num(v, "{:.2f}")),
    ("Temps d'entraînement (s)", "fit_seconds", _num),
]


def to_markdown(df: pd.DataFrame) -> str:
    out = ["# Comparaison LSTM (P6.3-P6.4) / Transformer (P6.5)", "",
           "Généré par `p6_models/scripts/p6_compare_seq.py` à partir des CSV de `p6_models/evaluation/`. "
           "Seuil fixé sur le bénin de validation à 0,1 % de FPR ; toutes les valeurs « test » sont mesurées "
           "sur le test, jamais utilisé pour choisir. Jitter : intervalles des sessions C2 multipliés par "
           "U(1 − j, 1 + j), bénin et seuil inchangés. « – » = CSV absent.", ""]
    if df.empty:
        return "\n".join(out + ["_Aucun résultat trouvé._", ""])
    for (base, group), d in df.groupby(["base_tag", "group"], sort=True):
        cols = []
        for arch in ARCHS:
            for _, r in d[d.arch == arch].sort_values("jitter_train").iterrows():
                name = "LSTM" if arch == "lstm" else "Transformer"
                if r["jitter_train"] > 0:
                    name += f" + jitter-train {round(100 * r['jitter_train'])} %"
                cols.append((name, r))
        out += [f"## {group} — {base.replace('_', ', ')}", "",
                "| | " + " | ".join(c[0] for c in cols) + " |",
                "|---|" + "---|" * len(cols)]
        for label, col, fmt in LINES:
            if col is None:
                vals = [f"[{_pct(r['test_recall_ci_low'])} ; {_pct(r['test_recall_ci_high'])}]"
                        if pd.notna(r.get("test_recall_ci_low")) else "–" for _, r in cols]
            else:
                vals = [fmt(r.get(col, np.nan)) for _, r in cols]
            out.append(f"| {label} | " + " | ".join(vals) + " |")
        n = d["test_n_c2"].dropna()
        if len(n):
            out.append(f"\n_{int(n.iloc[0]):,} sessions C2 dans le test._".replace(",", " "))
        out.append("")
    return "\n".join(out)


def main():
    df = collect()
    if df.empty:
        sys.exit(f"[!] aucun CSV lstm_*/transformer_*_metrics.csv dans {EVAL_DIR}")
    df.to_csv(CSV_OUT, index=False, float_format="%.6f")
    MD_OUT.write_text(to_markdown(df), encoding="utf-8")
    print(to_markdown(df))
    print(f"\n-> {CSV_OUT}\n-> {MD_OUT}")


if __name__ == "__main__":
    main()
