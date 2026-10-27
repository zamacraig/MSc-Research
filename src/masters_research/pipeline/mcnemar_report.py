"""McNemar significance-test report across the five candidate models.

Answers the question that comparative-F1 tables leave open: *given two
classifiers evaluated on identical samples, is the difference in their error
patterns real or attributable to chance?* Uses the paired non-parametric
McNemar test (with Yates continuity correction) plus a per-class breakdown
with Bonferroni-corrected p-values.

Inputs
------
Every model's held-out test predictions come from the ``test_probs.npz`` file
saved in its best-run directory: the graph models by :mod:`pipeline.train` and
the Random Forest / Keras MLP baselines by :mod:`pipeline.baselines_runner`,
which train inside the same experiment on the same features, partitions and
t+1 label. Nothing is retrained here, so every model's prediction vector is
aligned sample-by-sample and the baselines carry exactly the settings
reported in the thesis. The best run of each model is chosen on *validation*
F1, never on test F1, so the test set is not used for model selection.

Outputs
-------
- ``plots/mcnemar_predictions.npz``   — every model's per-sample predictions
  and the shared label vector, saved for reproducibility.
- ``plots/mcnemar_table.tex``         — a LaTeX-ready significance table
  covering the four thesis-critical pairwise comparisons.
- Console — the same table printed as plain text.

Standalone usage
----------------
    python -m masters_research.pipeline.mcnemar_report
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist


# ---------------------------------------------------------------------------
# GNN prediction loading
# ---------------------------------------------------------------------------

def _find_best_run_dir(models_dir: Path, model_type: str, df: pd.DataFrame) -> Optional[Path]:
    """Return the on-disk directory of the best run for a given model type,
    ranked by validation F1 (best_f1, then val_f1)."""
    sub = df[df["model_type"] == model_type].copy() if "model_type" in df.columns else df.copy()
    if sub.empty:
        return None
    for col in ("best_f1", "val_f1"):
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
    # Select on validation performance only; ranking on final_test_f1 would
    # let the held-out test set choose which run is reported.
    sort_cols = [c for c in ("best_f1", "val_f1") if c in sub.columns]
    sub = sub.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    best = sub.iloc[0]
    run_dir = models_dir / model_type / str(best["run_id"])
    return run_dir if run_dir.exists() else None


MODEL_FOLDERS = [
    ("RandomForest", "RandomForest"),
    ("KerasMLP", "KerasMLP"),
    ("SimpleGCN", "SimpleGCN"),
    ("TemporalGCN_GRU", "TemporalGCN_GRU"),
    ("TemporalAttentionGNN", "TemporalAttentionGNN"),
]


def _load_predictions(models_dir: Path, df: pd.DataFrame) -> dict:
    """Return {display_name: (labels, preds)} for each of the five models whose
    best-run has a test_probs.npz on disk. Labels must match across models
    (deterministic outer split) — an assertion is raised otherwise so the
    caller notices a split-config drift immediately."""
    out: dict = {}
    canonical_labels: Optional[np.ndarray] = None
    for display, folder in MODEL_FOLDERS:
        run_dir = _find_best_run_dir(models_dir, folder, df)
        if run_dir is None:
            print(f"[{display}] no best-run directory found — skipping.")
            continue
        npz_path = run_dir / "test_probs.npz"
        if not npz_path.exists():
            print(f"[{display}] test_probs.npz missing at {npz_path} — "
                  f"rerun training with the recent train.py to backfill.")
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            labels = np.asarray(data["labels"]).astype(int)
            probs = np.asarray(data["probs"])
        preds = probs.argmax(axis=1).astype(int)
        if canonical_labels is None:
            canonical_labels = labels
        elif not np.array_equal(labels, canonical_labels):
            raise ValueError(
                f"[{display}] label vector disagrees with the first-loaded model. "
                f"McNemar requires identical test samples across compared models. "
                f"Check that all best-runs used the same outer split and train_ratio."
            )
        out[display] = (labels, preds)
        print(f"[{display}] loaded {len(preds)} predictions from {run_dir.name}")
    return out


# ---------------------------------------------------------------------------
# McNemar computation
# ---------------------------------------------------------------------------

def _mcnemar_binary(preds_a: np.ndarray, preds_b: np.ndarray,
                    labels: np.ndarray) -> dict:
    """Classic McNemar test on paired correct/incorrect indicators.

    Contingency cells:
        a = both correct       b = A correct, B wrong
        c = A wrong, B correct d = both wrong

    Test statistic (with Yates continuity correction):
        chi2 = (|b - c| - 1)^2 / (b + c),  df = 1

    Returns a dict with b, c, chi2, p, and a plain-english direction of the
    disagreement so callers can label the significance direction correctly.
    """
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    denom = b + c
    if denom == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / denom
        p = float(1.0 - chi2_dist.cdf(chi2, df=1))
    if b > c:
        direction = "A more correct"
    elif c > b:
        direction = "B more correct"
    else:
        direction = "tie"
    return {"b": b, "c": c, "chi2": chi2, "p": p, "direction": direction}


def _mcnemar_per_class(preds_a: np.ndarray, preds_b: np.ndarray,
                       labels: np.ndarray, num_classes: int) -> list[dict]:
    """Run binary McNemar restricted to samples of each true class."""
    results = []
    for cls in range(num_classes):
        mask = (labels == cls)
        if mask.sum() == 0:
            continue
        results.append({
            "class": cls,
            "n_samples": int(mask.sum()),
            **_mcnemar_binary(preds_a[mask], preds_b[mask], labels[mask]),
        })
    return results


def _bonferroni_hold_out(results: list[dict], m: int) -> list[dict]:
    """In-place add of Bonferroni-corrected p-values to a list of test dicts."""
    for r in results:
        r["p_bonf"] = min(1.0, r["p"] * m)
    return results


# ---------------------------------------------------------------------------
# LaTeX table formatting
# ---------------------------------------------------------------------------

def _fmt_p(p: float) -> str:
    """LaTeX-friendly p-value formatter with the usual small-value convention."""
    if p < 0.001:
        return r"$<$ 0.001"
    return f"{p:.3f}"


def _to_latex_table(pair_results: list[dict], caption: str, label: str) -> str:
    """Build the LaTeX table string. One row per comparison, both the
    correct-vs-incorrect summary and its three per-class Bonferroni-adjusted
    p-values live on the same row for compactness."""
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{" + caption + r"}",
        r"    \label{" + label + r"}",
        r"    \resizebox{\textwidth}{!}{%",
        r"        \begin{tabular}{@{}lccccccc@{}}",
        r"            \hline",
        r"            \textbf{Comparison (A vs B)} & \textbf{$b$} & \textbf{$c$} & "
        r"\textbf{$\chi^2$} & \textbf{$p$} & "
        r"\textbf{$p_{\text{class 0}}$} & \textbf{$p_{\text{class 1}}$} & "
        r"\textbf{$p_{\text{class 2}}$} \\ \hline",
    ]
    for r in pair_results:
        overall = r["overall"]
        per_class_ps = {rr["class"]: rr["p_bonf"] for rr in r["per_class"]}
        row = (
            f"            {r['name']} & "
            f"{overall['b']} & {overall['c']} & "
            f"{overall['chi2']:.3f} & {_fmt_p(overall['p'])} & "
            f"{_fmt_p(per_class_ps.get(0, 1.0))} & "
            f"{_fmt_p(per_class_ps.get(1, 1.0))} & "
            f"{_fmt_p(per_class_ps.get(2, 1.0))} \\\\"
        )
        lines.append(row)
    lines.extend([r"            \hline", r"        \end{tabular}%", r"    }", r"\end{table}"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data-setup helpers — must mirror the training pipeline exactly
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(base_path: Path | str | None = None) -> Path:
    base_path = Path(base_path or Path(__file__).resolve().parents[3]).resolve()
    models_dir = base_path / "src" / "masters_research" / "artifacts" / "models"
    plots_dir = base_path / "src" / "masters_research" / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results_csv = models_dir / "grid_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"grid_results.csv missing at {results_csv}")
    df = pd.read_csv(results_csv)
    # Outer-CV fold runs (run_id contains "_f<k>_") have a different held-out set
    # per fold, so they cannot be paired with the single-split runs. The
    # cross-validated McNemar tests are produced separately by outer_cv.py.
    df = df[~df["run_id"].astype(str).str.contains(r"_f\d+_", regex=True)].copy()

    # --- Predictions for all five models (load from disk, no retraining) ---
    all_model_preds = _load_predictions(models_dir, df)
    if not all_model_preds:
        raise RuntimeError(
            "No test_probs.npz files were found. Run the training pipeline "
            "(graph models and baselines) to generate them."
        )
    missing = [name for name, _ in MODEL_FOLDERS if name not in all_model_preds]
    if missing:
        print(f"WARNING: no predictions found for {missing}; comparisons that need "
              f"them will be skipped.")

    # Canonical shared labels for the held-out test set every model used.
    canonical_labels = next(iter(all_model_preds.values()))[0]
    n_test = len(canonical_labels)
    num_classes = int(canonical_labels.max()) + 1
    print(f"Shared held-out test set: {n_test} samples, {num_classes} classes.\n")

    all_preds = {k: v[1] for k, v in all_model_preds.items()}

    # Persist every model's predictions plus the shared labels so any
    # downstream analysis can reproduce the significance tests from disk.
    np.savez(
        plots_dir / "mcnemar_predictions.npz",
        labels=canonical_labels,
        **all_preds,
    )
    print(f"\nSaved aligned predictions to {plots_dir / 'mcnemar_predictions.npz'}\n")

    # --- Run the four thesis-critical comparisons ---
    comparisons = [
        ("RF vs KerasMLP", "RandomForest", "KerasMLP"),
        ("RF vs SimpleGCN", "RandomForest", "SimpleGCN"),
        ("RF vs TemporalAttentionGNN", "RandomForest", "TemporalAttentionGNN"),
        ("TemporalAttentionGNN vs SimpleGCN", "TemporalAttentionGNN", "SimpleGCN"),
    ]
    pair_results = []
    for name, a, b in comparisons:
        if a not in all_preds or b not in all_preds:
            print(f"Skipping {name}: missing predictions.")
            continue
        overall = _mcnemar_binary(all_preds[a], all_preds[b], canonical_labels)
        per_class = _bonferroni_hold_out(
            _mcnemar_per_class(all_preds[a], all_preds[b], canonical_labels, num_classes),
            m=num_classes,
        )
        pair_results.append({"name": name, "overall": overall, "per_class": per_class})

    # --- Console summary ---
    print("\n" + "=" * 72)
    print("McNemar significance results (Yates-corrected; per-class p Bonferroni-adjusted)")
    print("=" * 72)
    header = f"{'Comparison':40s}  {'b':>4s} {'c':>4s} {'chi2':>7s} {'p':>10s}"
    print(header)
    for r in pair_results:
        o = r["overall"]
        p_str = "<0.001" if o["p"] < 0.001 else f"{o['p']:.3f}"
        print(f"{r['name']:40s}  {o['b']:>4d} {o['c']:>4d} {o['chi2']:>7.3f} {p_str:>10s}")

    # --- LaTeX table ---
    tex = _to_latex_table(
        pair_results,
        caption="McNemar significance tests across the four thesis-critical "
                "pairwise comparisons. $b$ and $c$ are the discordant "
                "contingency cells (A correct \\& B wrong, and vice versa); "
                "$\\chi^2$ uses the Yates continuity correction with one "
                "degree of freedom. Per-class $p$-values are Bonferroni "
                "corrected for three simultaneous tests.",
        label="tab:mcnemar-tests",
    )
    tex_path = plots_dir / "mcnemar_table.tex"
    tex_path.write_text(tex)
    print(f"\nLaTeX table saved to {tex_path}\n")
    print(tex)
    return tex_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    main(project_root)
