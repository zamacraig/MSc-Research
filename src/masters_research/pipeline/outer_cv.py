"""Rotating outer cross-validation: evaluate every month exactly once.

Why this exists
---------------
The ordinary pipeline holds out a single fixed 20% of months. That leaves ~190
labelled test nodes, and a weighted-F1 estimated on 190 samples carries a 95%
interval of roughly +/-0.07. Differences smaller than that -- which includes
every hyperparameter effect measured on this dataset so far -- cannot be
distinguished from noise, and a single split is additionally subject to the luck
of which months landed in it (observed spread across folds: 0.64 to 0.73).

This module rotates the interleaved hold-out across ``outer_cv_folds`` offsets.
Months are assigned to folds round-robin within each calendar month (see
create_temporal_dataset), so every fold sees a seasonally balanced slice and,
across all folds, every month is tested exactly once. The pooled evaluation
therefore covers the whole record
(~1200 labelled nodes rather than 190). The interval tightens by roughly the
square root of that ratio, to about +/-0.026.

Leak discipline
---------------
Each fold rebuilds the dataset from scratch with its own ``cv_fold``, so the
scaler, seasonal statistics, SPI reference period and target bin edges are refit
on that fold's trainval months alone. The outer split applies the identical fold
rule, so no month is ever both fitted on and tested on.

What "pooled" means
-------------------
Pooled predictions come from different fitted models (one per fold), so the
pooled score estimates the performance of the *procedure*, not of a single
trained model. That is the standard reading of a cross-validated score. The
folds share training data, so the pooled sample is not fully independent and the
interval below is a mild under-estimate of the true uncertainty; it is still far
closer to honest than the single-split figure.

Usage
-----
Set ``"outer_cv_folds": 5`` in the config and run the pipeline as normal.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import f1_score

from masters_research.pipeline.train import main as run_training, load_config


def _pool_by_model_type(models_dir: Path, run_prefix: str
                        ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Concatenate every fold's test predictions, keyed by architecture.

    Each run writes ``test_probs.npz`` into ``models/<ModelType>/<run_id>/``, and
    run ids carry an ``_f<fold>`` tag, so this picks up exactly the runs from this
    CV sweep and ignores unrelated historical runs in the same tree.

    Seeds are averaged WITHIN a fold before folds are concatenated, for two
    reasons. It mirrors how the models would actually be deployed (one ensembled
    prediction per month), and -- critically -- it makes every architecture emit
    exactly one prediction per labelled node even though they differ in run count
    (a 2x5 GNN grid produces 10 runs per fold, the baselines 5). Without it the
    pooled arrays would differ in length and every paired McNemar test would be
    silently skipped.
    """
    # (model_type, fold) -> {"labels": ndarray, "probs": [ndarray, ...]}
    buckets: dict[tuple[str, int], dict] = {}
    tag = re.compile(rf"^{re.escape(run_prefix)}_f(\d+)_")

    for npz_path in models_dir.rglob("test_probs.npz"):
        run_dir = npz_path.parent
        m = tag.match(run_dir.name)
        if not m:
            continue
        fold = int(m.group(1))
        model_type = run_dir.parent.name
        try:
            data = np.load(npz_path)
            labels = np.asarray(data["labels"]).astype(int)
            probs = np.asarray(data["probs"], dtype=float)
        except Exception:
            continue
        if labels.size == 0 or probs.size == 0:
            continue

        key = (model_type, fold)
        slot = buckets.setdefault(key, {"labels": labels, "probs": []})
        # Every run in a fold is scored on the same test nodes in the same order;
        # anything else means the run does not belong with these and is dropped.
        if slot["labels"].shape != labels.shape or not np.array_equal(slot["labels"], labels):
            continue
        slot["probs"].append(probs)

    by_model: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for (model_type, fold), slot in buckets.items():
        if not slot["probs"]:
            continue
        mean_probs = np.mean(np.stack(slot["probs"], axis=0), axis=0)
        by_model.setdefault(model_type, {})[fold] = (slot["labels"],
                                                     mean_probs.argmax(axis=1))

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_type, folds in by_model.items():
        order = sorted(folds)                      # fold order fixes row order
        y = np.concatenate([folds[f][0] for f in order])
        pred = np.concatenate([folds[f][1] for f in order])
        out[model_type] = (y, pred)
    return out


def _mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float, float]:
    """McNemar with Yates continuity correction — matches mcnemar_report.py."""
    n10 = int((correct_a & ~correct_b).sum())
    n01 = int((~correct_a & correct_b).sum())
    if (n10 + n01) == 0:
        return n10, n01, 0.0, 1.0
    stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    return n10, n01, float(stat), float(chi2.sf(stat, 1))


def main(base_path: Path, config_path: Path | None = None) -> None:
    base_path = Path(base_path).resolve()
    if config_path is None:
        config_path = base_path / "configs" / "default.json"
    cfg = load_config(Path(config_path).resolve())

    n_folds = int(cfg.get("outer_cv_folds", 5))
    run_prefix = cfg.get("run_name", "run")

    print("=" * 66)
    print(f"ROTATING OUTER CROSS-VALIDATION — {n_folds} folds")
    print("Every month is held out exactly once; preprocessing is refit per fold.")
    print("=" * 66)

    for fold in range(n_folds):
        print(f"\n{'=' * 66}\nFOLD {fold + 1} of {n_folds}\n{'=' * 66}")
        run_training(
            base_path=base_path,
            config_path=Path(config_path),
            cv_fold=fold,
            n_outer_folds=n_folds,
            run_reports=False,      # aggregates are produced once, below
        )

    # ---------------- pooling ----------------
    default_models = base_path / "src" / "masters_research" / "artifacts" / "models"
    models_dir = Path(cfg.get("save_dir", default_models))
    models_dir = models_dir if models_dir.is_absolute() else (base_path / models_dir)

    pooled = _pool_by_model_type(models_dir, run_prefix)
    if not pooled:
        print("\nNo pooled predictions found — nothing to summarise.")
        return

    print(f"\n{'=' * 66}\nPOOLED RESULTS ACROSS {n_folds} FOLDS\n{'=' * 66}")
    rows = []
    for model_type, (y, pred) in sorted(
        pooled.items(), key=lambda kv: -f1_score(kv[1][0], kv[1][1], average="weighted")
    ):
        f1 = f1_score(y, pred, average="weighted")
        acc = float((y == pred).mean())
        # Normal-approximation interval on accuracy, used as the resolution
        # guide for the F1 estimated on the same sample.
        se = float(np.sqrt(acc * (1 - acc) / len(y)))
        rows.append({
            "model_type": model_type, "pooled_n": len(y),
            "pooled_weighted_f1": round(f1, 4), "pooled_accuracy": round(acc, 4),
            "ci95_halfwidth": round(1.96 * se, 4),
        })
        print(f"  {model_type:<24} n={len(y):<5} F1={f1:.4f}  acc={acc:.4f}  "
              f"95% CI +/-{1.96 * se:.4f}")

    out_csv = models_dir / f"{run_prefix}_outer_cv_pooled.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nPooled summary written to: {out_csv}")

    # ---------------- pairwise McNemar ----------------
    # Only comparable where two architectures were scored on an identical
    # pooled sample; differing lengths mean different fold coverage.
    names = [r["model_type"] for r in rows]
    print(f"\n{'=' * 66}\nPAIRWISE McNEMAR (pooled, Yates-corrected)\n{'=' * 66}")
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ya, pa = pooled[a]
            yb, pb = pooled[b]
            if len(ya) != len(yb) or not np.array_equal(ya, yb):
                print(f"  {a} vs {b}: skipped (different pooled samples)")
                continue
            n10, n01, stat, p = _mcnemar(pa == ya, pb == yb)
            verdict = "significant" if p < 0.05 else "not significant"
            print(f"  {a:<22} vs {b:<22} {a[:12]}-only={n10:<4} "
                  f"{b[:12]}-only={n01:<4} chi2={stat:6.2f} p={p:.4f}  {verdict}")
            pairs.append({"model_a": a, "model_b": b, "a_only_correct": n10,
                          "b_only_correct": n01, "chi2": round(stat, 4),
                          "p_value": round(p, 6), "significant_at_0.05": p < 0.05})

    if pairs:
        pair_csv = models_dir / f"{run_prefix}_outer_cv_mcnemar.csv"
        pd.DataFrame(pairs).to_csv(pair_csv, index=False)
        print(f"\nPairwise tests written to: {pair_csv}")
