"""Station-level results: weighted F1 of every model at each station.

Answers whether the graph models' shortfall is spread across the network or
driven by a few stations. One row per station with the number of held-out
test cases, the class mix, each model's weighted F1, and the gap between the
non-graph baselines and the graph models.

It works on the per-case table written by ``error_analysis.py``
(``<run>_error_analysis_cases.csv``), which holds the pooled (seed-averaged,
fold-concatenated) prediction of every model for every held-out
(station, month) case. The scores are therefore the same predictions as the
pooled results and McNemar tests, split by station; the "All stations" row
reproduces the pooled weighted F1.

Each station has only about 110-140 test cases, so a station's F1 is
uncertain by roughly +/-0.08. Read the table for the pattern across stations,
not for small differences at one station.

Outputs (in the models directory, prefixed with the run name):
    <run>_station_f1.csv   the table
    <run>_station_f1.txt   the same table, readable
    <run>_station_f1.tex   LaTeX tabular for the dissertation
    <run>_station_f1.png   dot plot, one row per station

Standalone usage:
    python -m masters_research.pipeline.station_report --config configs/spi3_cv.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

MODEL_ORDER = ["RandomForest", "KerasMLP", "SimpleGCN", "TemporalGCN_GRU", "TemporalAttentionGNN"]
BASELINES = ("RandomForest", "KerasMLP")
SHORT = {"RandomForest": "RF", "KerasMLP": "MLP", "SimpleGCN": "SimpleGCN",
         "TemporalGCN_GRU": "GCN-GRU", "TemporalAttentionGNN": "Attention"}


def station_names(base_path: Path) -> dict[int, str]:
    """Station number -> short name from StationsData.csv (first entry per number)."""
    path = Path(base_path) / "Dataset" / "OriginalDataset" / "StationsData.csv"
    try:
        st = pd.read_csv(path, sep=";", skiprows=[0], header=None,
                         names=["id", "name", "lat", "lon", "alt"])
    except Exception:
        return {}
    st = st.dropna(subset=["id"]).drop_duplicates("id")
    # "PRIESKA; BLAAUKRANS" -> "Prieska"
    return {int(r.id): str(r.name).split(";")[0].split(":")[0].strip().title() for r in st.itertuples()}


def _f1(y, p) -> float:
    return float(f1_score(y, p, average="weighted")) if len(y) else float("nan")


def station_table(cases: pd.DataFrame, models: list[str], names: dict[int, str] | None = None) -> pd.DataFrame:
    names = names or {}
    base = [m for m in models if m in BASELINES]
    graph = [m for m in models if m not in BASELINES]
    rows = []
    groups = [(s, sub) for s, sub in cases.groupby("station")] + [("All stations", cases)]
    for station, sub in groups:
        row = {
            "station": station,
            "name": names.get(int(station), "") if station != "All stations" else "",
            "n": len(sub),
            "n_dry": int((sub.label == 0).sum()),
            "n_middle": int((sub.label == 1).sum()),
            "n_wet": int((sub.label == 2).sum()),
        }
        for m in models:
            row[m] = round(_f1(sub.label, sub[m]), 4)
        if base and graph:
            row["baseline_mean"] = round(float(np.mean([row[m] for m in base])), 4)
            row["graph_mean"] = round(float(np.mean([row[m] for m in graph])), 4)
            row["gap"] = round(row["baseline_mean"] - row["graph_mean"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _to_tex(t: pd.DataFrame, models: list[str]) -> str:
    cols = ["Station", "$n$"] + [SHORT.get(m, m) for m in models] + (["Gap"] if "gap" in t else [])
    lines = [r"\begin{tabular}{l r " + "r " * (len(cols) - 2) + "}", r"\toprule",
             " & ".join(cols) + r" \\", r"\midrule"]
    for _, r in t.iterrows():
        if r.station == "All stations":
            lines.append(r"\midrule")
            label = "All stations"
        else:
            label = f"{r['name']} ({r.station})" if r["name"] else str(r.station)
        vals = [f"{r[m]:.3f}" for m in models] + ([f"{r.gap:.3f}"] if "gap" in t else [])
        lines.append(" & ".join([label, str(int(r.n))] + vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _plot(t: pd.DataFrame, models: list[str], out_path: Path, title: str) -> None:
    t = t[t.station != "All stations"].copy()
    t["label"] = [f"{r['name']} ({r.station})\nn={int(r.n)}" if r["name"] else f"{r.station}\nn={int(r.n)}"
                  for _, r in t.iterrows()]
    t = t.sort_values("gap" if "gap" in t else models[0])
    colors = {"RandomForest": "#1b6ca8", "KerasMLP": "#5fa8d3", "SimpleGCN": "#c0392b",
              "TemporalGCN_GRU": "#e67e22", "TemporalAttentionGNN": "#8e44ad"}
    markers = {"RandomForest": "o", "KerasMLP": "s", "SimpleGCN": "^",
               "TemporalGCN_GRU": "v", "TemporalAttentionGNN": "D"}
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(t) + 1.6))
    y = np.arange(len(t))
    for m in models:
        ax.scatter(t[m], y, label=SHORT.get(m, m), color=colors.get(m), marker=markers.get(m), s=46, zorder=3)
    for i, (_, r) in enumerate(t.iterrows()):
        vals = [r[m] for m in models]
        ax.plot([min(vals), max(vals)], [i, i], color="0.8", linewidth=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(t["label"], fontsize=8)
    ax.set_xlabel("Weighted F1 on held-out cases")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(models), fontsize=8, frameon=False)
    ax.set_title(title, fontsize=10, pad=24)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_station_report(cases: pd.DataFrame, models: list[str], models_dir: Path, run_prefix: str,
                         base_path: Path | None = None) -> Path:
    """Build and save the station table from a per-case prediction table."""
    models = [m for m in MODEL_ORDER if m in models]
    names = station_names(base_path) if base_path is not None else {}
    t = station_table(cases, models, names)
    prefix = Path(models_dir) / f"{run_prefix}_station_f1"
    t.to_csv(f"{prefix}.csv", index=False)
    Path(f"{prefix}.tex").write_text(_to_tex(t, models), encoding="utf-8")
    header = (f"STATION-LEVEL WEIGHTED F1  ({run_prefix}, {len(cases)} held-out cases)\n"
              "gap = mean of non-graph baselines minus mean of graph models.\n"
              "Each station has ~110-140 cases: a station's F1 is uncertain by about +/-0.08.\n\n")
    Path(f"{prefix}.txt").write_text(header + t.to_string(index=False) + "\n", encoding="utf-8")
    try:
        _plot(t, models, Path(f"{prefix}.png"), f"Weighted F1 by station ({run_prefix})")
    except Exception as e:
        print(f"  station plot skipped: {e}")

    print(f"\n{'=' * 66}\nSTATION-LEVEL WEIGHTED F1\n{'=' * 66}")
    print(t.drop(columns=["n_dry", "n_middle", "n_wet"]).to_string(index=False))
    print(f"Station table written to: {prefix}.csv / .txt / .tex / .png")
    return Path(f"{prefix}.csv")


def main(base_path: Path, config_path: Path | None = None) -> Path | None:
    from masters_research.pipeline.train import load_config
    base_path = Path(base_path).resolve()
    config_path = Path(config_path) if config_path else base_path / "configs" / "default.json"
    cfg = load_config(config_path.resolve())
    models_dir = Path(cfg.get("save_dir", base_path / "src" / "masters_research" / "artifacts" / "models"))
    models_dir = models_dir if models_dir.is_absolute() else base_path / models_dir
    run_prefix = cfg.get("run_name", "run")
    cases_csv = models_dir / f"{run_prefix}_error_analysis_cases.csv"
    if not cases_csv.exists():
        print(f"{cases_csv} not found; run error_analysis for this config first.")
        return None
    cases = pd.read_csv(cases_csv)
    models = [m for m in MODEL_ORDER if m in cases.columns]
    return write_station_report(cases, models, models_dir, run_prefix, base_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", default=None)
    a = ap.parse_args()
    main(Path(a.base) if a.base else Path(__file__).resolve().parents[3], a.config)
