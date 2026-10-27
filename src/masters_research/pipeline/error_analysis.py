"""Error analysis of the held-out predictions.

Accuracy and per-class scores say how often a model is wrong; this module says
how it is wrong and where the errors sit. It is run at the end of the pipeline
(``outer_cv.py`` for the rotating cross-validation, ``train.py`` for a single
split) on the predictions that were already saved, and it can also be run on its
own.

For every held-out (station, month) case it rebuilds the context the saved
``test_probs.npz`` files do not carry -- the fold, the calendar month, the
station and the class of the *current* month (the class a no-model "persistence"
rule would repeat) -- and then reports, per model:

  * error types: adjacent-class errors against dry<->wet (outer-class) reversals,
    and where the near-typical class goes;
  * the dry<->wet reversals by direction, by station, fold and season, and
    whether they follow genuine changes of class;
  * accuracy when the class stays the same as the current month against when it
    changes, next to the persistence rule (a diagnostic, not a trained model);
  * accuracy by calendar season and by station;
  * per-run behaviour: which runs collapsed onto one or two classes, and whether
    the errors come from those runs;
  * the cases every model gets wrong, and those where only one family does.

Classes follow the pipeline: 0 driest tercile, 1 near typical, 2 wettest. The
analysis assumes three ordered classes (the dry<->wet reversal is class 0 <-> 2).

Station numbers are the node indices of the graph, which are the sorted station
numbers (``LabelEncoder`` order in ``create_temporal_dataset``).

Outputs (in the models directory, prefixed with the run name):
    <run>_error_analysis_cases.csv      one row per held-out case, with each model's prediction
    <run>_error_analysis_types.csv      error types per model
    <run>_error_analysis_outer.csv      dry<->wet reversal rate by station / fold / season, with p-values
    <run>_error_analysis_change.csv     accuracy by class change, with the persistence rule
    <run>_error_analysis_season.csv     accuracy by season
    <run>_error_analysis_station.csv    accuracy by station
    <run>_error_analysis_fold.csv       accuracy by fold
    <run>_error_analysis_runs.csv       per-run class shares, accuracy and reversals
    <run>_error_analysis_collapse.csv   collapsed against non-collapsed runs, per model
    <run>_error_analysis_shared.csv     cases wrong for all models / one family only
    <run>_error_analysis.txt            the same results as readable text

The counts are descriptive. Station-months are dependent (neighbouring stations,
consecutive months, overlapping SPI windows), so the chi-square p-values are a
guide to whether a difference exceeds chance variation, not formal tests.

Standalone usage:
    python -m masters_research.pipeline.error_analysis --config configs/spi3_cv.json
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2_contingency
from sklearn.metrics import f1_score
from torch_geometric_temporal.signal import temporal_signal_split

from masters_research.pipeline.dataset import create_temporal_dataset, interleaved_temporal_split

MODEL_ORDER = ["RandomForest", "KerasMLP", "SimpleGCN", "TemporalGCN_GRU", "TemporalAttentionGNN"]
BASELINES = ("RandomForest", "KerasMLP")
SEASONS = {12: "Dec-Feb", 1: "Dec-Feb", 2: "Dec-Feb", 3: "Mar-May", 4: "Mar-May", 5: "Mar-May",
           6: "Jun-Aug", 7: "Jun-Aug", 8: "Jun-Aug", 9: "Sep-Nov", 10: "Sep-Nov", 11: "Sep-Nov"}
# A run is "collapsed" if it (almost) never predicts the near-typical class, or
# if a single class takes most of its predictions.
COLLAPSE_MIN_MIDDLE_SHARE = 0.05
COLLAPSE_MAX_CLASS_SHARE = 0.60


# ---------------------------------------------------------------------------
# Loading saved predictions
# ---------------------------------------------------------------------------

def _load_runs(models_dir: Path, run_prefix: str, n_folds: int) -> list[dict]:
    """Every saved run of this experiment: model type, fold, labels, probabilities."""
    fold_tag = re.compile(rf"^{re.escape(run_prefix)}_f(\d+)_")
    plain_tag = re.compile(rf"^{re.escape(run_prefix)}(?!_f\d+_)")
    runs = []
    for npz in sorted(models_dir.rglob("test_probs.npz")):
        name = npz.parent.name
        if n_folds > 1:
            m = fold_tag.match(name)
            if not m:
                continue
            fold = int(m.group(1)) + 1
        else:
            if not plain_tag.match(name):
                continue
            fold = 1
        try:
            z = np.load(npz)
            labels, probs = np.asarray(z["labels"]).astype(int), np.asarray(z["probs"], dtype=float)
        except Exception:
            continue
        if labels.size == 0 or probs.size == 0:
            continue
        runs.append(dict(model=npz.parent.parent.name, fold=fold, run=name, labels=labels, probs=probs))
    return runs


def _ensemble(runs: list[dict], n_folds: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One prediction per case and model: probabilities averaged over the runs of
    a fold (seeds, and widths for the graph models), folds concatenated in order.
    This is the same pooling as ``outer_cv._pool_by_model_type``."""
    buckets: dict[tuple[str, int], dict] = {}
    for r in runs:
        slot = buckets.setdefault((r["model"], r["fold"]), dict(labels=r["labels"], probs=[]))
        if slot["labels"].shape == r["labels"].shape and np.array_equal(slot["labels"], r["labels"]):
            slot["probs"].append(r["probs"])
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in sorted({m for m, _ in buckets}):
        folds = sorted(f for m, f in buckets if m == model)
        y = np.concatenate([buckets[(model, f)]["labels"] for f in folds])
        pred = np.concatenate([np.mean(buckets[(model, f)]["probs"], axis=0).argmax(1) for f in folds])
        out[model] = (y, pred)
    return out


# ---------------------------------------------------------------------------
# Case context (fold, month, station, current class)
# ---------------------------------------------------------------------------

def _to_np(x) -> np.ndarray:
    return (x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x))


def _month_of(ts) -> pd.Timestamp:
    """Calendar month of a snapshot timestamp (12 h added: month starts carry a UTC offset)."""
    sec = float(_to_np(ts).reshape(-1)[0])
    t = pd.Timestamp(sec, unit="s") + pd.Timedelta(hours=12)
    return pd.Timestamp(t.year, t.month, 1)


def _case_table(base_path: Path, cfg: dict, n_folds: int) -> pd.DataFrame:
    """Fold, month, node, label and current-month class of every held-out case,
    in the order of the pooled predictions (folds, then snapshots, then nodes)."""
    data_path = base_path / cfg.get("data_path", "Dataset/TransientData/processedData.csv")
    df = pd.read_csv(data_path)
    holdout = float(cfg.get("test_holdout_ratio", 0.2))
    stratified = bool(cfg.get("stratified_split", False))
    rows = []
    for fold in range(max(n_folds, 1)):
        cv = fold if n_folds > 1 else None
        ds = create_temporal_dataset(
            df,
            seasonal_normalization=cfg.get("seasonal_normalization", True),
            add_lag_features=cfg.get("add_lag_features", True),
            n_classes=int(cfg.get("num_classes", 3)),
            test_holdout_ratio=holdout, stratified_split=stratified,
            forecast_horizon=int(cfg.get("forecast_horizon", 1)),
            target_type=str(cfg.get("target_type", "feature_mean")),
            target_window=int(cfg.get("target_window", 3)),
            base_path=base_path, cv_fold=cv, n_outer_folds=n_folds,
            purge_heldout_lags=bool(cfg.get("purge_heldout_lags", False)),
        )
        # Label of every snapshot's forecast month, so the label of the PREVIOUS
        # snapshot is the class of the current month.
        by_month = {_month_of(t): _to_np(y).astype(int) for t, y in zip(ds.timestamps, ds.targets)}
        if n_folds > 1 or stratified:
            _, test = interleaved_temporal_split(ds, train_ratio=1.0 - holdout, cv_fold=cv, n_outer_folds=n_folds)
        else:
            _, test = temporal_signal_split(ds, train_ratio=1.0 - holdout)
        for i in range(test.snapshot_count):
            y = _to_np(test.targets[i]).astype(int)
            month = _month_of(test.timestamps[i])
            prev = by_month.get(month - pd.DateOffset(months=1))
            for node in np.flatnonzero(y >= 0):
                cur = int(prev[node]) if prev is not None and node < len(prev) and prev[node] >= 0 else -1
                rows.append(dict(fold=fold + 1, month=month, node=int(node), label=int(y[node]), cur=cur))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analyses (all take the case table with one prediction column per model)
# ---------------------------------------------------------------------------

def _p_chi2(groups: pd.Series, flag: pd.Series) -> float:
    ct = pd.crosstab(groups, flag)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    return float(chi2_contingency(ct)[1])


def _reversal(d: pd.DataFrame, m: str) -> pd.Series:
    """True where a dry case was predicted wet or a wet case dry."""
    return ((d.label == 0) & (d[m] == 2)) | ((d.label == 2) & (d[m] == 0))


def error_types(d: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    outer = d.label.isin([0, 2])
    mid = d.label == 1
    rows = []
    for m in models:
        wrong = d[m] != d.label
        rev = _reversal(d, m)
        rows.append(dict(
            model=m, n=len(d), errors=int(wrong.sum()), accuracy=round(1 - wrong.mean(), 4),
            adjacent_errors=int((wrong & ~rev).sum()), dry_wet_errors=int(rev.sum()),
            dry_wet_share_of_errors=round(rev.sum() / max(wrong.sum(), 1), 4),
            dry_wet_rate_among_outer_cases=round(rev[outer].mean(), 4),
            dry_predicted_wet=int(((d.label == 0) & (d[m] == 2)).sum()),
            wet_predicted_dry=int(((d.label == 2) & (d[m] == 0)).sum()),
            middle_cases=int(mid.sum()),
            middle_sent_to_outer_share=round(float((d[m][mid] != 1).mean()), 4),
            predicted_middle_share=round(float((d[m] == 1).mean()), 4)))
    return pd.DataFrame(rows)


def outer_reversals(d: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Dry<->wet reversal rate among truly dry or wet cases, by station, fold and
    season, with a chi-square p-value for "the rate differs between groups"."""
    o = d[d.label.isin([0, 2])].copy()
    o["season"] = o.month.dt.month.map(SEASONS)
    rows = []
    for key in ("node", "fold", "season"):
        for m in models:
            flag = _reversal(o, m)
            p = _p_chi2(o[key], flag)
            for g, sub in o.groupby(key):
                rows.append(dict(by=key, group=g, model=m, n_outer_cases=len(sub),
                                 reversal_rate=round(float(flag[sub.index].mean()), 4), p_value_across_groups=round(p, 4)))
    return pd.DataFrame(rows)


def by_change(d: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Accuracy when the class stays the same as the current month and when it changes."""
    v = d[d.cur >= 0].copy()
    v["changed"] = v.cur != v.label
    rows = [dict(model="Persistence rule", all_cases=round((v.cur == v.label).mean(), 4),
                 unchanged=1.0, changed=0.0, predicts_current_class=1.0)]
    for m in models:
        ok = v[m] == v.label
        rows.append(dict(model=m, all_cases=round(ok.mean(), 4), unchanged=round(ok[~v.changed].mean(), 4),
                         changed=round(ok[v.changed].mean(), 4),
                         predicts_current_class=round((v[m] == v.cur).mean(), 4)))
    out = pd.DataFrame(rows)
    out["n_all"], out["n_unchanged"], out["n_changed"] = len(v), int((~v.changed).sum()), int(v.changed.sum())
    return out


def by_group(d: pd.DataFrame, models: list[str], key: str) -> pd.DataFrame:
    """Accuracy per model by season / station / fold, with the share of cases whose
    class changed and the persistence accuracy where the current class is known."""
    d = d.copy()
    d["season"] = d.month.dt.month.map(SEASONS)
    rows = []
    for g, sub in d.groupby(key):
        known = sub[sub.cur >= 0]
        row = dict(group=g, n=len(sub),
                   class_change_share=round(float((known.cur != known.label).mean()), 4) if len(known) else np.nan,
                   persistence_accuracy=round(float((known.cur == known.label).mean()), 4) if len(known) else np.nan)
        row.update({m: round(float((sub[m] == sub.label).mean()), 4) for m in models})
        rows.append(row)
    return pd.DataFrame(rows)


def run_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        pred = r["probs"].argmax(1)
        y = r["labels"]
        share = [float((pred == c).mean()) for c in range(3)]
        rows.append(dict(
            model=r["model"], fold=r["fold"], run=r["run"], n=len(y), accuracy=round(float((pred == y).mean()), 4),
            weighted_f1=round(float(f1_score(y, pred, average="weighted")), 4),
            share_dry=round(share[0], 4), share_middle=round(share[1], 4), share_wet=round(share[2], 4),
            dry_wet_errors=int((((y == 0) & (pred == 2)) | ((y == 2) & (pred == 0))).sum()),
            collapsed=bool(share[1] < COLLAPSE_MIN_MIDDLE_SHARE or max(share) > COLLAPSE_MAX_CLASS_SHARE)))
    return pd.DataFrame(rows)


def collapse_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, c), sub in runs_df.groupby(["model", "collapsed"]):
        rows.append(dict(model=m, collapsed=c, n_runs=len(sub), mean_accuracy=round(sub.accuracy.mean(), 4),
                         mean_weighted_f1=round(sub.weighted_f1.mean(), 4), mean_middle_share=round(sub.share_middle.mean(), 4),
                         dry_wet_error_rate=round(sub.dry_wet_errors.sum() / sub.n.sum(), 4),
                         mean_dry_wet_errors_per_run=round(sub.dry_wet_errors.mean(), 2),
                         max_dry_wet_errors_per_run=int(sub.dry_wet_errors.max())))
    return pd.DataFrame(rows)


def shared_failures(d: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    wrong = {m: d[m] != d.label for m in models}
    base = [m for m in models if m in BASELINES]
    graph = [m for m in models if m not in BASELINES]
    rows = []

    def add(name, mask):
        rows.append(dict(group=name, cases=int(mask.sum()), share=round(float(mask.mean()), 4),
                         dry=int((mask & (d.label == 0)).sum()), middle=int((mask & (d.label == 1)).sum()),
                         wet=int((mask & (d.label == 2)).sum())))

    add("all models wrong", np.logical_and.reduce([wrong[m] for m in models]))
    if base and graph:
        gw = np.logical_and.reduce([wrong[m] for m in graph])
        gr = np.logical_and.reduce([~wrong[m] for m in graph])
        bw = np.logical_and.reduce([wrong[m] for m in base])
        br = np.logical_and.reduce([~wrong[m] for m in base])
        add("all graph models wrong, all baselines right", gw & br)
        add("all baselines wrong, all graph models right", bw & gr)
        rev = np.array([_reversal(d, m).values for m in graph])
        outer = d.label.isin([0, 2]).values
        counts = np.bincount(rev[:, outer].sum(0), minlength=len(graph) + 1)
        for k, c in enumerate(counts):
            rows.append(dict(group=f"outer-class cases reversed by {k} of {len(graph)} graph models", cases=int(c),
                             share=round(float(c / max(outer.sum(), 1)), 4), dry=np.nan, middle=np.nan, wet=np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _write_report(path: Path, run_prefix: str, n_folds: int, tables: dict[str, pd.DataFrame], n_cases: int) -> None:
    mode = f"{n_folds}-fold rotating outer cross-validation" if n_folds > 1 else "single split"
    lines = [f"ERROR ANALYSIS  ({run_prefix}, {mode}, {n_cases} held-out cases)", "=" * 72, ""]
    for name, t in tables.items():
        lines += [name.upper().replace("_", " "), "-" * 72, t.to_string(index=False), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_error_analysis(base_path: Path, cfg: dict, models_dir: Path, run_prefix: str, n_folds: int) -> Path | None:
    """Run the full analysis on the saved predictions of one experiment."""
    base_path, models_dir = Path(base_path), Path(models_dir)
    # Grid-style list fields are irrelevant here; only the dataset settings are read.
    cfg = {k: (v[0] if isinstance(v, list) and v else v) for k, v in cfg.items()}
    runs = _load_runs(models_dir, run_prefix, n_folds)
    pooled = _ensemble(runs, n_folds)
    models = [m for m in MODEL_ORDER if m in pooled]
    if not models:
        print("Error analysis skipped: no saved predictions found.")
        return None

    cases = _case_table(base_path, cfg, n_folds)
    labels = pooled[models[0]][0]
    if len(cases) != len(labels) or not np.array_equal(cases.label.values, labels):
        print("Error analysis skipped: rebuilt held-out cases do not match the saved labels "
              "(was the config or dataset changed after training?).")
        return None
    for m in models:
        if pooled[m][0].shape != labels.shape or not np.array_equal(pooled[m][0], labels):
            print(f"  {m}: predictions are on a different case set; left out of the error analysis.")
        else:
            cases[m] = pooled[m][1]
    models = [m for m in models if m in cases.columns]

    # Station numbers are the sorted station ids (LabelEncoder order).
    stations = sorted(pd.read_csv(base_path / cfg.get("data_path", "Dataset/TransientData/processedData.csv"),
                                  usecols=["Compno_x"]).Compno_x.unique())
    cases["station"] = cases.node.map(lambda n: stations[n] if n < len(stations) else n)

    runs_df = run_table([r for r in runs if r["model"] in models])
    tables = {
        "types": error_types(cases, models),
        "outer": outer_reversals(cases, models),
        "change": by_change(cases, models),
        "season": by_group(cases, models, "season"),
        "station": by_group(cases, models, "station"),
        "fold": by_group(cases, models, "fold"),
        "runs": runs_df,
        "collapse": collapse_summary(runs_df),
        "shared": shared_failures(cases, models),
    }
    prefix = models_dir / f"{run_prefix}_error_analysis"
    cases.to_csv(f"{prefix}_cases.csv", index=False)
    for name, t in tables.items():
        t.to_csv(f"{prefix}_{name}.csv", index=False)
    report = Path(f"{prefix}.txt")
    _write_report(report, run_prefix, n_folds, {k: v for k, v in tables.items() if k not in ("runs", "outer")}, len(cases))

    t = tables["types"]
    print(f"\n{'=' * 66}\nERROR ANALYSIS ({len(cases)} held-out cases)\n{'=' * 66}")
    for _, r in t.iterrows():
        print(f"  {r.model:<22} errors={r.errors:<4} dry<->wet={r.dry_wet_errors:<4} "
              f"({100 * r.dry_wet_share_of_errors:.0f}% of errors)  middle sent to outer={100 * r.middle_sent_to_outer_share:.0f}%")
    print(f"Error analysis written to: {report}")

    try:
        from masters_research.pipeline.station_report import write_station_report
        write_station_report(cases, models, models_dir, run_prefix, base_path)
    except Exception as e:
        print(f"Station-level report skipped: {e}")
    return report


def main(base_path: Path, config_path: Path | None = None) -> Path | None:
    from masters_research.pipeline.train import load_config
    base_path = Path(base_path).resolve()
    config_path = Path(config_path) if config_path else base_path / "configs" / "default.json"
    cfg = load_config(config_path.resolve())
    default_models = base_path / "src" / "masters_research" / "artifacts" / "models"
    models_dir = Path(cfg.get("save_dir", default_models))
    models_dir = models_dir if models_dir.is_absolute() else base_path / models_dir
    return run_error_analysis(base_path, cfg, models_dir, cfg.get("run_name", "run"), int(cfg.get("outer_cv_folds", 1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", default=None)
    a = ap.parse_args()
    main(Path(a.base) if a.base else Path(__file__).resolve().parents[3], a.config)
