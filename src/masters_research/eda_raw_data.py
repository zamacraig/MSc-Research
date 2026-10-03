"""
Exploratory Data Analysis -- raw weather-station data and final modelling dataset.

Run from the project root:
    python src/masters_research/eda_raw_data.py

Covers:
  1. Features present in each raw file (schema consistency)
  2. Period covered and record count per station
  3. Monthly-record breakdown per station (records per year)
  4. Missingness per variable per station
  5. Final modelling-data balance (class and year distribution)
  6. Station-coverage summary table + geographic map
  7. Seasonality -- monthly climatology, inter-annual anomalies,
     autocorrelation, and seasonal distribution by month
"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
matplotlib.rcParams.update({"figure.dpi": 130, "font.size": 9})

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "Dataset" / "OriginalDataset"
PROC_DIR = PROJECT_ROOT / "Dataset" / "TransientData"
OUT_DIR = PROJECT_ROOT / "src" / "masters_research" / "artifacts" / "plots" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = [
    "St_BrandvleiB.csv",
    "St_Kakamas.csv",
    "St_Luckhof.csv",
    "St_Olifantshoek.csv",
    "St_Prieska.csv",
    "St_Vaalharts.csv",
    "St_VanWyksvleiB.csv",
    "St_BerkleyWest.csv",
]

FEATURE_COLS = ["Tx", "Tn", "RHx", "RHn", "Rain", "Rs", "U2", "ET0"]
FEATURE_LABELS = {
    "Tx":   "Max Temp",
    "Tn":   "Min Temp",
    "RHx":  "Max Humidity",
    "RHn":  "Min Humidity",
    "Rain": "Rainfall",
    "Rs":   "Radiation",
    "U2":   "Wind Speed",
    "ET0":  "Evaporation",
}

PALETTE = sns.color_palette("tab10")

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── helpers ────────────────────────────────────────────────────────────────────
def _banner(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def _hline() -> None:
    print("-" * 70)


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.relative_to(PROJECT_ROOT)}")


def _short_name(fname: str) -> str:
    return fname.replace("St_", "").replace(".csv", "")


def _load_raw(path: Path) -> pd.DataFrame:
    """Read one station CSV; tilde -> NaN; decimal comma -> period."""
    df = pd.read_csv(path, sep=";", header=0, dtype=str)
    df = df.replace(r"~+", np.nan, regex=True)
    df = df.replace(",", ".", regex=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def _load_station_meta() -> pd.DataFrame:
    meta = pd.read_csv(
        RAW_DIR / "StationsData.csv",
        sep=";",
        skiprows=[0],
        header=None,
        names=["Compno", "StationName", "Latitude", "Longitude", "Altitude"],
        dtype=str,
    )
    meta = meta.replace(",", ".", regex=True).replace(r"~+", np.nan, regex=True)
    meta["Compno"] = pd.to_numeric(meta["Compno"], errors="coerce")
    meta = meta.dropna(subset=["Compno"])
    meta["Compno"] = meta["Compno"].astype(int)
    meta = meta.drop_duplicates(subset=["Compno"])
    return meta.reset_index(drop=True)


# ── section 1: schema consistency ─────────────────────────────────────────────
def analyse_schema(all_files: dict) -> None:
    _banner("1. SCHEMA CONSISTENCY -- features present in each raw file")

    schema_rows = []
    for fname, df in sorted(all_files.items()):
        cols = [c for c in df.columns if c not in ("Compno", "Year", "Month")]
        schema_rows.append({"File": fname, "Columns": ", ".join(cols)})

    schema_df = pd.DataFrame(schema_rows)
    print(schema_df.to_string(index=False))

    expected = set(FEATURE_COLS)
    print()
    any_diff = False
    for _, row in schema_df.iterrows():
        actual = set(row["Columns"].split(", "))
        diff = actual.symmetric_difference(expected)
        if diff:
            print(f"  [NOTE] {row['File']} diverges -- extra/missing: {diff}")
            any_diff = True
    if not any_diff:
        print("  All model-pipeline files share identical feature columns.")

    schemas = {
        f: tuple(sorted(c for c in df.columns if c not in ("Compno", "Year", "Month")))
        for f, df in all_files.items()
        if f in MODEL_FILES
    }
    unique_schemas = set(schemas.values())
    print(f"\n  Unique feature schemas across the 8 model files: {len(unique_schemas)}")
    if len(unique_schemas) == 1:
        print("  [OK] All 8 model-pipeline files have identical feature sets.")
    else:
        for schema in unique_schemas:
            files = [f for f, s in schemas.items() if s == schema]
            print(f"    Schema {schema}  ->  {files}")


def plot_schema(all_files: dict) -> None:
    """Binary heatmap: every file x every possible feature."""
    all_features = sorted(
        {c for df in all_files.values() for c in df.columns if c not in ("Compno", "Year", "Month")}
    )
    rows = []
    for fname, df in sorted(all_files.items()):
        present = {c for c in df.columns if c not in ("Compno", "Year", "Month")}
        rows.append({f: int(f in present) for f in all_features})

    matrix = pd.DataFrame(rows, index=sorted(all_files.keys()))
    matrix.index = [_short_name(i) for i in matrix.index]

    # Shade model-pipeline files differently
    model_mask = pd.DataFrame(False, index=matrix.index, columns=matrix.columns)
    for fname in MODEL_FILES:
        short = _short_name(fname)
        if short in model_mask.index:
            model_mask.loc[short] = True

    fig, ax = plt.subplots(figsize=(9, 4))
    cmap = matplotlib.colors.ListedColormap(["#f0f0f0", "#2196F3"])
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        annot=matrix.applymap(lambda x: "yes" if x else ""),
        fmt="",
        annot_kws={"size": 7},
    )
    # Highlight model-file rows with a left border marker
    for i, row_name in enumerate(matrix.index):
        if row_name in [_short_name(f) for f in MODEL_FILES]:
            ax.add_patch(
                plt.Rectangle((-0.18, i), 0.15, 1, color="#2ecc71", clip_on=False, lw=0)
            )
    ax.set_title("Feature schema by file  (green bar = used in modelling pipeline)", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, "01_schema_matrix.png")


# ── section 2: period coverage and record count ────────────────────────────────
def analyse_coverage(raw_files: dict, meta: pd.DataFrame) -> pd.DataFrame:
    _banner("2. PERIOD COVERED AND RECORD COUNT PER STATION")

    rows = []
    for fname in MODEL_FILES:
        df = raw_files.get(fname)
        if df is None:
            continue
        compno = int(df["Compno"].dropna().iloc[0])
        station_name = meta.loc[meta["Compno"] == compno, "StationName"].values
        station_name = station_name[0].strip() if len(station_name) else "Unknown"

        ts = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1), errors="coerce")
        n_total = len(df)
        n_valid_ts = int(ts.notna().sum())
        start = ts.min()
        end = ts.max()

        if pd.notna(start) and pd.notna(end):
            expected = (end.year - start.year) * 12 + (end.month - start.month) + 1
        else:
            expected = np.nan

        rows.append(
            {
                "File": fname,
                "Compno": compno,
                "Station": station_name,
                "Start": start.strftime("%Y-%m") if pd.notna(start) else "N/A",
                "End": end.strftime("%Y-%m") if pd.notna(end) else "N/A",
                "Start_dt": start,
                "End_dt": end,
                "Records": n_total,
                "Expected_months": int(expected) if not np.isnan(expected) else "N/A",
                "Missing_months": (
                    int(expected - n_valid_ts) if not np.isnan(expected) else "N/A"
                ),
            }
        )

    cov_df = pd.DataFrame(rows)
    display_cols = [c for c in cov_df.columns if c not in ("Start_dt", "End_dt")]
    print(cov_df[display_cols].to_string(index=False))
    return cov_df


def plot_coverage_timeline(cov_df: pd.DataFrame) -> None:
    """Gantt chart of temporal coverage per station."""
    df = cov_df.dropna(subset=["Start_dt", "End_dt"]).copy()
    df = df.sort_values("Start_dt")
    labels = [_short_name(f) for f in df["File"]]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = sns.color_palette("Set2", len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        ax.barh(
            i,
            (row["End_dt"] - row["Start_dt"]).days,
            left=row["Start_dt"],
            height=0.6,
            color=colors[i],
            edgecolor="white",
            linewidth=0.4,
        )
        ax.text(
            row["End_dt"],
            i,
            f"  {row['Records']} records",
            va="center",
            fontsize=7.5,
            color="#333",
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlabel("Year")
    ax.set_title("Temporal coverage per station (width = date range, label = record count)")
    fig.tight_layout()
    _save(fig, "02_coverage_timeline.png")


# ── section 3: monthly records per station ────────────────────────────────────
def analyse_monthly_records(raw_files: dict) -> None:
    _banner("3. MONTHLY RECORDS PER STATION -- records per calendar year")

    for fname in MODEL_FILES:
        df = raw_files.get(fname)
        if df is None:
            continue
        compno = int(df["Compno"].dropna().iloc[0])
        by_year = df.groupby("Year").size().rename("N_records")
        by_year.index = by_year.index.astype(int)

        flags = {yr: ("  << incomplete" if n < 12 else "") for yr, n in by_year.items()}
        print(f"\n  {fname}  (Compno {compno})")
        _hline()
        for yr, n in by_year.items():
            bar = "#" * n
            print(f"    {int(yr)}  {n:3d}  {bar}{flags[yr]}")


def plot_monthly_heatmap(raw_files: dict) -> None:
    """Heatmap: station x year, coloured by number of records (0-12)."""
    records = {}
    for fname in MODEL_FILES:
        df = raw_files.get(fname)
        if df is None:
            continue
        by_year = df.groupby("Year").size()
        by_year.index = by_year.index.astype(int)
        records[_short_name(fname)] = by_year

    matrix = pd.DataFrame(records).T.sort_index(axis=1)
    matrix = matrix.fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(13, 4))
    cmap = sns.color_palette("YlGn", as_cmap=True)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=12,
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt="d",
        annot_kws={"size": 7},
        cbar_kws={"label": "N records", "shrink": 0.7},
    )
    ax.set_title("Monthly records per station per year  (max = 12, grey = no data)")
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, "03_monthly_coverage_heatmap.png")


# ── section 4: missingness per variable ───────────────────────────────────────
def analyse_missingness(raw_files: dict) -> pd.DataFrame:
    _banner("4. MISSINGNESS PER VARIABLE PER STATION  (% NaN after tilde -> NaN)")

    miss_rows = []
    for fname in MODEL_FILES:
        df = raw_files.get(fname)
        if df is None:
            continue
        compno = int(df["Compno"].dropna().iloc[0])
        n = len(df)
        row = {"File": fname, "Compno": compno, "N_rows": n}
        for col in FEATURE_COLS:
            if col in df.columns:
                pct = df[col].isna().mean() * 100
                row[col] = f"{pct:.1f}%"
            else:
                row[col] = "N/A"
        miss_rows.append(row)

    miss_df = pd.DataFrame(miss_rows)
    print(miss_df.to_string(index=False))

    print("\n  Overall missingness across all 8 model files:")
    combined = pd.concat(
        [raw_files[f] for f in MODEL_FILES if f in raw_files], ignore_index=True
    )
    overall = {}
    for col in FEATURE_COLS:
        if col in combined.columns:
            pct = combined[col].isna().mean() * 100
            overall[col] = f"{pct:.1f}%"
    print("  " + "  ".join(f"{k}: {v}" for k, v in overall.items()))
    return miss_df


def plot_missingness(raw_files: dict) -> None:
    """Heatmap of % missing per feature per station."""
    numeric_rows = []
    for fname in MODEL_FILES:
        df = raw_files.get(fname)
        if df is None:
            continue
        row = {col: df[col].isna().mean() * 100 if col in df.columns else np.nan
               for col in FEATURE_COLS}
        numeric_rows.append(row)

    miss_matrix = pd.DataFrame(
        numeric_rows,
        index=[_short_name(f) for f in MODEL_FILES if f in raw_files],
    )
    miss_matrix.columns = [FEATURE_LABELS.get(c, c) for c in miss_matrix.columns]

    fig, ax = plt.subplots(figsize=(9, 4))
    cmap = sns.color_palette("Reds", as_cmap=True)
    sns.heatmap(
        miss_matrix,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=max(miss_matrix.max().max(), 1),
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt=".1f",
        annot_kws={"size": 8},
        cbar_kws={"label": "% missing", "shrink": 0.7},
    )
    ax.set_title("Missingness (%) per variable per station  (tildes treated as NaN)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, "04_missingness_heatmap.png")


# ── section 5: final modelling data balance ────────────────────────────────────
def analyse_modelling_balance() -> pd.DataFrame | None:
    _banner("5. FINAL MODELLING-DATA BALANCE  (processedData.csv)")

    proc_path = PROC_DIR / "processedData.csv"
    if not proc_path.exists():
        print(f"  [SKIP] {proc_path} not found.")
        return None

    proc = pd.read_csv(proc_path)
    nodes = proc[["TimeStamp", "Compno_x", "Target_x"]].drop_duplicates().copy()
    ts = nodes["TimeStamp"]
    if pd.api.types.is_numeric_dtype(ts):
        nodes["Year"] = pd.to_datetime(ts, unit="s").dt.year
    else:
        nodes["Year"] = pd.to_datetime(ts).dt.year
    nodes["Target_x"] = nodes["Target_x"].astype(int)

    n_stations = nodes["Compno_x"].nunique()
    n_timestamps = nodes["TimeStamp"].nunique()
    print(f"  Unique stations (encoded): {n_stations}")
    print(f"  Unique timestamps:         {n_timestamps}")
    print(f"  Node-level rows:           {len(nodes)}")

    print("\n  Records per station (encoded Compno_x):")
    station_counts = nodes["Compno_x"].value_counts().sort_index()
    for sid, cnt in station_counts.items():
        bar = "#" * int(cnt / max(station_counts) * 30)
        print(f"    Station {sid:2d}  {cnt:4d}  {bar}")

    print("\n  Records per year:")
    year_counts = nodes["Year"].value_counts().sort_index()
    for yr, cnt in year_counts.items():
        bar = "#" * int(cnt / max(year_counts) * 30)
        print(f"    {yr}  {cnt:4d}  {bar}")

    print("\n  Target-class distribution per station (row = station, col = class):")
    pivot = nodes.groupby(["Compno_x", "Target_x"]).size().unstack(fill_value=0)
    pivot.index.name = "Station"
    pivot.columns = [f"Class_{c}" for c in pivot.columns]
    print(pivot.to_string())

    print("\n  Overall class distribution:")
    overall_cls = nodes["Target_x"].value_counts().sort_index()
    total = overall_cls.sum()
    for cls, cnt in overall_cls.items():
        bar = "#" * int(cnt / total * 40)
        print(f"    Class {cls}  {cnt:4d}  ({cnt/total*100:5.1f}%)  {bar}")

    return nodes


def plot_modelling_balance(nodes: pd.DataFrame) -> None:
    """Four-panel figure for the modelling dataset balance."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Final modelling dataset balance  (processedData.csv)", fontsize=11, y=1.01)

    # ── (a) records per station ────────────────────────────────────────────────
    ax = axes[0, 0]
    station_counts = nodes["Compno_x"].value_counts().sort_index()
    bars = ax.bar(
        [f"S{i}" for i in station_counts.index],
        station_counts.values,
        color=sns.color_palette("tab10", len(station_counts)),
        edgecolor="white",
    )
    ax.axhline(station_counts.mean(), color="red", linestyle="--", linewidth=0.8, label="mean")
    ax.bar_label(bars, fontsize=7, padding=2)
    ax.set_title("(a) Records per station")
    ax.set_xlabel("Encoded station ID")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    ax.set_ylim(0, station_counts.max() * 1.15)

    # ── (b) records per year ───────────────────────────────────────────────────
    ax = axes[0, 1]
    year_counts = nodes["Year"].value_counts().sort_index()
    color_yr = ["#e74c3c" if cnt < year_counts.median() else "#2196F3"
                for cnt in year_counts.values]
    bars = ax.bar(year_counts.index.astype(str), year_counts.values,
                  color=color_yr, edgecolor="white")
    ax.bar_label(bars, fontsize=7, padding=2)
    ax.axhline(year_counts.median(), color="orange", linestyle="--",
               linewidth=0.8, label="median")
    ax.set_title("(b) Records per year  (red = below median)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, year_counts.max() * 1.15)

    # ── (c) class distribution per station (heatmap) ──────────────────────────
    ax = axes[1, 0]
    pivot = nodes.groupby(["Compno_x", "Target_x"]).size().unstack(fill_value=0)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="Blues",
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt="d",
        annot_kws={"size": 8},
        cbar_kws={"label": "count", "shrink": 0.7},
    )
    ax.set_title("(c) Target-class distribution per station")
    ax.set_xlabel("Target class")
    ax.set_ylabel("Station (encoded)")
    ax.tick_params(axis="y", rotation=0)

    # ── (d) overall class balance (bar + expected line) ────────────────────────
    ax = axes[1, 1]
    overall_cls = nodes["Target_x"].value_counts().sort_index()
    total = overall_cls.sum()
    expected_per_class = total / len(overall_cls)
    bars = ax.bar(
        [f"Class {c}" for c in overall_cls.index],
        overall_cls.values,
        color=sns.color_palette("Set2", len(overall_cls)),
        edgecolor="white",
    )
    ax.axhline(expected_per_class, color="red", linestyle="--",
               linewidth=0.8, label=f"perfect balance ({expected_per_class:.0f})")
    ax.bar_label(bars, labels=[f"{v/total*100:.1f}%" for v in overall_cls.values],
                 fontsize=7, padding=2)
    ax.set_title("(d) Overall target-class distribution")
    ax.set_xlabel("Target class")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    ax.set_ylim(0, overall_cls.max() * 1.2)

    fig.tight_layout()
    _save(fig, "05_modelling_balance.png")


# ── section 6: station-coverage summary table + map ───────────────────────────
def station_summary_table(cov_df: pd.DataFrame, meta: pd.DataFrame) -> None:
    _banner("6. STATION COVERAGE SUMMARY TABLE")

    meta_slim = meta[["Compno", "Latitude", "Longitude", "Altitude"]].copy()
    summary = cov_df.merge(meta_slim, on="Compno", how="left")

    cols_ordered = [
        "Compno", "Station", "Latitude", "Longitude", "Altitude",
        "Start", "End", "Records", "Expected_months", "Missing_months",
    ]
    summary = summary[[c for c in cols_ordered if c in summary.columns]]

    pd.set_option("display.max_colwidth", 30)
    pd.set_option("display.width", 120)
    print(summary.to_string(index=False))

    print("\n  Legend:")
    print("    Records         - total rows in the raw CSV")
    print("    Expected_months - months between Start and End (inclusive)")
    print("    Missing_months  - gap months with no record (Expected - actual)")


def plot_station_map(cov_df: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Scatter map of station locations coloured by total records."""
    meta_slim = meta[["Compno", "Latitude", "Longitude", "Altitude"]].copy()
    for col in ("Latitude", "Longitude", "Altitude"):
        meta_slim[col] = pd.to_numeric(meta_slim[col], errors="coerce")

    summary = cov_df.merge(meta_slim, on="Compno", how="left")
    valid = summary.dropna(subset=["Latitude", "Longitude"])
    missing_geo = summary[summary["Latitude"].isna()]

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        valid["Longitude"],
        valid["Latitude"],
        c=valid["Records"],
        cmap="YlOrRd",
        s=180,
        edgecolors="#333",
        linewidth=0.6,
        zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Total records")

    for _, row in valid.iterrows():
        name = _short_name(row["File"]) if "File" in row else str(row["Compno"])
        ax.annotate(
            name,
            xy=(row["Longitude"], row["Latitude"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.5,
            color="#222",
        )

    if not missing_geo.empty:
        note = ", ".join(_short_name(r["File"]) for _, r in missing_geo.iterrows())
        ax.text(
            0.02, 0.04, f"No coords: {note}",
            transform=ax.transAxes, fontsize=7, color="grey",
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Station locations  (colour = total records in raw CSV)")
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save(fig, "06_station_map.png")


def plot_records_summary(cov_df: pd.DataFrame) -> None:
    """Stacked bar: actual vs missing months per station."""
    df = cov_df.copy()
    df["Label"] = df["File"].apply(_short_name)
    df["Missing"] = pd.to_numeric(df["Missing_months"], errors="coerce").fillna(0).astype(int)
    df["Present"] = df["Records"]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df))
    w = 0.55
    ax.bar(x, df["Present"], width=w, label="Records present", color="#2196F3", edgecolor="white")
    ax.bar(x, df["Missing"], width=w, bottom=df["Present"],
           label="Gap months", color="#e74c3c", alpha=0.7, edgecolor="white")

    for i, row in df.iterrows():
        idx = list(df.index).index(i)
        ax.text(idx, row["Present"] + row["Missing"] + 1, str(row["Missing"]),
                ha="center", va="bottom", fontsize=7, color="#c0392b")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"], rotation=20, ha="right")
    ax.set_ylabel("Months")
    ax.set_title("Records present vs gap months per station  (red number = gaps)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "07_records_vs_gaps.png")


# ── section 7: seasonality helpers ───────────────────────────────────────────
def _build_combined(raw_files: dict, meta: pd.DataFrame) -> pd.DataFrame:
    """Stack all model station files into one DataFrame with ShortName labels."""
    frames = []
    for fname in MODEL_FILES:
        df = raw_files.get(fname)
        if df is None:
            continue
        cols = ["Compno", "Year", "Month"] + [c for c in FEATURE_COLS if c in df.columns]
        d = df[cols].copy()
        compno = int(d["Compno"].dropna().iloc[0])
        match = meta.loc[meta["Compno"] == compno, "StationName"].values
        d["ShortName"] = _short_name(fname)
        d["Station"] = match[0].strip() if len(match) else _short_name(fname)
        frames.append(d)
    combined = pd.concat(frames, ignore_index=True)
    for col in FEATURE_COLS:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["Month"] = combined["Month"].astype(int)
    combined["Year"] = combined["Year"].astype(int)
    return combined


def _acf(x: np.ndarray, max_lag: int = 36) -> np.ndarray:
    """Autocorrelation function up to max_lag, ignoring NaNs."""
    x = x[~np.isnan(x)]
    x = x - x.mean()
    n = len(x)
    var = np.var(x)
    if var == 0 or n < max_lag + 2:
        return np.array([1.0] + [0.0] * max_lag)
    result = np.empty(max_lag + 1)
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        result[lag] = np.dot(x[lag:], x[:-lag]) / (n * var)
    return result


# ── section 7: seasonality analysis ──────────────────────────────────────────
def analyse_seasonality(combined: pd.DataFrame) -> None:
    _banner("7. SEASONALITY -- monthly climatology across all stations")

    monthly = combined.groupby("Month")[FEATURE_COLS].mean()
    monthly.index = MONTH_NAMES
    print("\n  Monthly climatological means (pooled across all stations and years):")
    print(monthly.round(2).to_string())

    print("\n  Seasonal signal per feature  (peak / trough / absolute range):")
    header = f"    {'Feature':22s}  {'Peak':5s}  {'Trough':6s}  {'Range':>8s}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for col in FEATURE_COLS:
        if col not in monthly.columns:
            continue
        peak_m   = monthly[col].idxmax()
        trough_m = monthly[col].idxmin()
        rng      = monthly[col].max() - monthly[col].min()
        label    = FEATURE_LABELS.get(col, col)
        print(f"    {label:22s}  {peak_m:5s}  {trough_m:6s}  {rng:8.2f}")


def plot_monthly_climatology(combined: pd.DataFrame) -> None:
    """2x4 grid: monthly mean per feature with per-station lines and pooled mean/std band."""
    stations = sorted(combined["ShortName"].unique())
    colors = dict(zip(stations, sns.color_palette("tab10", len(stations))))

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    axes = axes.flatten()

    for ax, col in zip(axes, FEATURE_COLS):
        for sname in stations:
            sub = combined[combined["ShortName"] == sname]
            monthly_st = sub.groupby("Month")[col].mean()
            ax.plot(monthly_st.index, monthly_st.values,
                    color=colors[sname], alpha=0.6, linewidth=1.1, label=sname)

        # Pooled mean ± 1 std band
        gp = combined.groupby("Month")[col]
        mu  = gp.mean()
        sig = gp.std()
        ax.plot(mu.index, mu.values, color="black", linewidth=2,
                linestyle="--", zorder=5, label="All-stn mean")
        ax.fill_between(mu.index, mu - sig, mu + sig, color="grey", alpha=0.15)

        ax.set_title(FEATURE_LABELS.get(col, col), fontsize=9)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], fontsize=7)
        ax.grid(linestyle="--", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.03), frameon=True)
    fig.suptitle(
        "Monthly climatology per variable  (coloured lines = stations, dashed = pooled mean, shading = ±1 std)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, "08_monthly_climatology.png")


def plot_seasonal_anomaly(combined: pd.DataFrame) -> None:
    """Year x Month z-score anomaly heatmap for 4 key variables."""
    KEY_VARS = ["Tx", "Tn", "Rain", "ET0"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, col in zip(axes, KEY_VARS):
        frames = []
        for sname in combined["ShortName"].unique():
            sub = combined[combined["ShortName"] == sname][["Year", "Month", col]].dropna(subset=[col])
            # Per-station monthly climatological mean and std
            clim_mean = sub.groupby("Month")[col].mean()
            clim_std  = sub.groupby("Month")[col].std().clip(lower=0.001)
            sub = sub.copy()
            sub["anom"] = (sub[col] - sub["Month"].map(clim_mean)) / sub["Month"].map(clim_std)
            frames.append(sub[["Year", "Month", "anom"]])

        anom_all   = pd.concat(frames, ignore_index=True)
        # Average anomaly across stations for each (Year, Month)
        pivot = anom_all.groupby(["Year", "Month"])["anom"].mean().unstack()
        pivot.columns = [MONTH_NAMES[int(m) - 1] for m in pivot.columns]

        abs_max = max(pivot.abs().max().max(), 0.1)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "z-score", "shrink": 0.7},
            annot=(len(pivot) <= 20),
            fmt=".1f",
            annot_kws={"size": 6},
        )
        ax.set_title(f"{FEATURE_LABELS.get(col, col)}  (red = above, blue = below climatology)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "Seasonal anomalies: z-score relative to station-specific monthly climatology, averaged across stations",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, "09_seasonal_anomaly.png")


def plot_autocorrelation(combined: pd.DataFrame) -> None:
    """2x4 ACF plots: mean ACF across stations per feature, confirming 12-month cycle."""
    MAX_LAG = 36

    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    axes = axes.flatten()

    for ax, col in zip(axes, FEATURE_COLS):
        station_acfs = []
        n_total = 0
        for sname in combined["ShortName"].unique():
            sub = (
                combined[combined["ShortName"] == sname]
                .sort_values(["Year", "Month"])
                [col]
                .values
            )
            acf_vals = _acf(sub, max_lag=MAX_LAG)
            station_acfs.append(acf_vals)
            n_total += (~np.isnan(sub)).sum()

        mean_acf = np.mean(station_acfs, axis=0)
        lags = np.arange(len(mean_acf))
        sig  = 1.96 / np.sqrt(n_total / len(station_acfs))  # approx 95% CI

        bar_colors = ["#e74c3c" if abs(v) > sig else "#bdc3c7" for v in mean_acf]
        ax.bar(lags, mean_acf, color=bar_colors, width=0.85, zorder=2)
        ax.axhline(sig,  color="#2c3e50", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.axhline(-sig, color="#2c3e50", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.4)

        for lag_mark, alpha in [(12, 1.0), (24, 0.5), (36, 0.3)]:
            ax.axvline(lag_mark, color="darkorange", linestyle=":", linewidth=1.0, alpha=alpha)

        ax.set_title(FEATURE_LABELS.get(col, col), fontsize=9)
        ax.set_xlabel("Lag (months)", fontsize=7)
        ax.set_xlim(-0.5, MAX_LAG + 0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(linestyle="--", alpha=0.2)

    # Shared legend for lag markers
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="darkorange", linestyle=":", label="lag 12 / 24 / 36"),
        Line2D([0], [0], color="#2c3e50", linestyle="--", label="95% CI"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle(
        "Autocorrelation function per variable  (mean across stations; red bars = significant; orange = annual multiples)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "10_autocorrelation.png")


def plot_seasonal_boxplots(combined: pd.DataFrame) -> None:
    """2x4 box plots: value distribution per calendar month for each feature."""
    plot_data = combined.copy()
    plot_data["MonthName"] = plot_data["Month"].apply(lambda m: MONTH_NAMES[m - 1])

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()
    palette = sns.color_palette("coolwarm", 12)

    for ax, col in zip(axes, FEATURE_COLS):
        sns.boxplot(
            data=plot_data.dropna(subset=[col]),
            x="MonthName",
            y=col,
            order=MONTH_NAMES,
            ax=ax,
            palette=palette,
            linewidth=0.7,
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.4},
        )
        ax.set_title(FEATURE_LABELS.get(col, col), fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        "Seasonal distribution by month  (pooled across all stations and years)",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, "11_seasonal_boxplots.png")


def plot_inter_annual_trend(combined: pd.DataFrame) -> None:
    """Annual mean per station per variable — shows long-term drift alongside seasonal variation."""
    annual = combined.groupby(["ShortName", "Year"])[FEATURE_COLS].mean().reset_index()
    stations = sorted(annual["ShortName"].unique())
    colors = dict(zip(stations, sns.color_palette("tab10", len(stations))))

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()

    for ax, col in zip(axes, FEATURE_COLS):
        for sname in stations:
            sub = annual[annual["ShortName"] == sname].sort_values("Year")
            if sub[col].notna().sum() < 2:
                continue
            ax.plot(sub["Year"], sub[col], marker="o", markersize=3,
                    linewidth=1.0, color=colors[sname], alpha=0.75, label=sname)

        # Overall annual trend (pooled)
        pool = combined.groupby("Year")[col].mean()
        ax.plot(pool.index, pool.values, color="black", linewidth=2,
                linestyle="--", zorder=5, label="All-stn mean")

        ax.set_title(FEATURE_LABELS.get(col, col), fontsize=9)
        ax.set_xlabel("")
        ax.grid(linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.03), frameon=True)
    fig.suptitle(
        "Annual mean per variable per station  (dashed = pooled mean; shows inter-annual trend)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, "12_inter_annual_trend.png")


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\nWeather-Station Exploratory Data Analysis")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw data     : {RAW_DIR}")
    print(f"Processed    : {PROC_DIR}")
    print(f"Figures out  : {OUT_DIR}")

    all_files: dict = {}
    for fp in sorted(RAW_DIR.glob("St_*.csv")):
        all_files[fp.name] = _load_raw(fp)

    meta = _load_station_meta()
    raw_files = {k: v for k, v in all_files.items() if k in MODEL_FILES}

    analyse_schema(all_files)
    plot_schema(all_files)

    cov_df = analyse_coverage(raw_files, meta)
    plot_coverage_timeline(cov_df)
    plot_records_summary(cov_df)

    analyse_monthly_records(raw_files)
    plot_monthly_heatmap(raw_files)

    analyse_missingness(raw_files)
    plot_missingness(raw_files)

    combined = _build_combined(raw_files, meta)
    analyse_seasonality(combined)
    plot_monthly_climatology(combined)
    plot_seasonal_anomaly(combined)
    plot_autocorrelation(combined)
    plot_seasonal_boxplots(combined)
    plot_inter_annual_trend(combined)

    nodes = analyse_modelling_balance()
    if nodes is not None:
        plot_modelling_balance(nodes)

    station_summary_table(cov_df, meta)
    plot_station_map(cov_df, meta)

    print("\n" + "=" * 70)
    print(f"  EDA complete. Figures saved to: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
