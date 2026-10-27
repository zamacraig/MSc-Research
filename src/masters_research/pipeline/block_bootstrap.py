"""Dependence-aware comparison of the pooled outer-CV predictions.

Station-months are spatially and temporally dependent (neighbouring stations,
consecutive months, overlapping SPI-3 windows), so McNemar p-values on the pooled
predictions are optimistic. This module resamples whole calendar months in
circular blocks (all stations of a month stay together) and recomputes the paired
accuracy and weighted-F1 difference of every model pair on the saved predictions.

Output: ``<run_name>_outer_cv_block_bootstrap.csv`` in the models directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from masters_research.pipeline.outer_cv import _pool_by_model_type

ORDER = ["RandomForest", "KerasMLP", "SimpleGCN", "TemporalGCN_GRU", "TemporalAttentionGNN"]


def _weighted_f1(cm: np.ndarray) -> float:
    sup = cm.sum(1)
    tp = np.diag(cm)
    pr = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    rc = np.divide(tp, sup, out=np.zeros(3), where=sup > 0)
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros(3), where=(pr + rc) > 0)
    return float((f * sup).sum() / sup.sum())


# Dataset settings of the primary SPI-3 cross-validation, used when no config is given.
DEFAULT_CFG = dict(seasonal_normalization=True, add_lag_features=True, num_classes=3, test_holdout_ratio=0.2,
                   stratified_split=True, forecast_horizon=1, target_type="spi", target_window=3)


def _case_months(root: Path, n_folds: int = 5, cfg: dict | None = None) -> tuple[pd.Series, np.ndarray]:
    """Calendar month and label of every pooled case, in pooled (fold) order."""
    from masters_research.pipeline.error_analysis import _case_table
    cases = _case_table(Path(root), cfg or DEFAULT_CFG, n_folds)
    return cases["month"], cases["label"].to_numpy()


def main(base_path: Path | str | None = None, run_name: str = "spi3cv",
         blocks=(1, 3, 6), n_boot: int = 10000, seed: int = 20261019,
         cfg: dict | None = None, models_dir: Path | str | None = None, n_folds: int = 5) -> Path:
    """Block bootstrap of every model pair on the saved pooled outer-CV predictions.

    Called by ``outer_cv.py`` with the run's config (``bootstrap_blocks``,
    ``bootstrap_resamples`` and ``bootstrap_seed`` override the defaults there),
    or standalone with the defaults of the primary experiment."""
    root = Path(base_path) if base_path else Path(__file__).resolve().parents[3]
    md = Path(models_dir) if models_dir else root / "src/masters_research/artifacts/models"
    cfg = {k: (v[0] if isinstance(v, list) and v else v) for k, v in (cfg or DEFAULT_CFG).items()}
    months, labels = _case_months(root, n_folds, cfg)
    pooled = _pool_by_model_type(md, run_name)
    order = [m for m in ORDER if m in pooled]
    for m in order:
        assert np.array_equal(pooled[m][0], labels), "reconstructed labels differ from the saved labels"
    preds = {m: pooled[m][1] for m in order}

    uniq = sorted(months.unique())
    pos = {t: i for i, t in enumerate(uniq)}
    mpos = months.map(pos).values
    n_months = len(uniq)

    def stats(w):
        out = {}
        for m in order:
            cm = np.bincount(labels * 3 + preds[m], weights=w, minlength=9).reshape(3, 3)
            out[m] = (np.trace(cm) / cm.sum(), _weighted_f1(cm))
        return out

    pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    rng = np.random.default_rng(seed)
    obs = stats(np.ones(len(labels)))
    rows = []
    for block in blocks:
        res = {p: ([], []) for p in pairs}
        for _ in range(n_boot):
            st = rng.integers(0, n_months, size=int(np.ceil(n_months / block)))
            idx = ((st[:, None] + np.arange(block)[None, :]) % n_months).ravel()[:n_months]
            s = stats(np.bincount(idx, minlength=n_months).astype(float)[mpos])
            for a, b in pairs:
                res[(a, b)][0].append(s[a][0] - s[b][0])
                res[(a, b)][1].append(s[a][1] - s[b][1])
        for a, b in pairs:
            for k, name in ((0, "accuracy"), (1, "weighted_f1")):
                d = np.array(res[(a, b)][k])
                lo, hi = np.percentile(d, [2.5, 97.5])
                lo2, hi2 = np.percentile(d, [0.25, 99.75])
                p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                rows.append(dict(block=block, model_a=a, model_b=b, metric=name, diff=obs[a][k] - obs[b][k],
                                 ci95_lo=lo, ci95_hi=hi, ci995_lo=lo2, ci995_hi=hi2, p_boot=max(p, 1 / n_boot)))
    out = md / f"{run_name}_outer_cv_block_bootstrap.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    print(main())
