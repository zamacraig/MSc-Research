import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch
import numpy as np


def _parse_timestamps(values) -> pd.Series:
    """Parse a TimeStamp column into proper datetimes, regardless of storage format.

    The transient dataset has been produced in two different formats over the
    life of this project: ISO date strings ("2004-11-01") and Unix epoch
    *seconds* stored as floats (1099267200.0). A bare ``pd.to_datetime`` call
    silently mis-reads the latter as **nanoseconds**, mapping every row to
    1970-01-01 — which collapses all rows into a single calendar month and
    destroys both the seasonal normalisation and the Fourier encodings without
    raising an error. This helper detects the numeric case and applies
    ``unit="s"`` so both formats resolve to the same calendar dates.
    """
    s = pd.Series(values)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="s")
    return pd.to_datetime(s)


def _compute_seasonal_stats(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Compute per-month (mean, std) statistics for the requested feature columns.

    Kept separate from `_apply_seasonal_normalization` so the caller can fit
    these statistics on the trainval slice only and then apply them to the
    entire dataframe — the shape needed to prevent test-set values from
    influencing the "typical January" baseline used to normalise training
    features (see the leak-free pipeline design in Chapter 4 of the thesis).
    """
    tmp = df.copy()
    tmp["_month"] = _parse_timestamps(tmp["TimeStamp"]).dt.month
    return tmp.groupby("_month")[features].agg(["mean", "std"])


def _apply_seasonal_normalization(
    df: pd.DataFrame,
    features: list[str],
    stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Normalize features relative to per-month statistics.

    This converts raw values to z-scores within each month, so that:
    - A hot summer day (35°C in July) → z ≈ 0 (normal for July)
    - A cold winter day (10°C in January) → z ≈ 0 (normal for January)

    Parameters
    ----------
    df, features
        As before — the frame to transform and the columns to normalise.
    stats
        Pre-computed per-month (mean, std) DataFrame as returned by
        `_compute_seasonal_stats`. If None, stats are computed from the
        input dataframe itself — legacy behaviour, retained for backward
        compatibility with any caller that doesn't supply a trainval slice.
    """
    df = df.copy()

    # Extract month from timestamp for the lookup below.
    df["_month"] = _parse_timestamps(df["TimeStamp"]).dt.month

    # Fit-on-supplied path (leak-free): use caller-provided stats.
    # Fallback path: compute over the input df (mirrors the pre-fix behaviour).
    if stats is None:
        stats = df.groupby("_month")[features].agg(["mean", "std"])

    # Z-score normalize each feature within its month.
    for feat in features:
        month_mean = df["_month"].map(stats[(feat, "mean")])
        month_std = df["_month"].map(stats[(feat, "std")]).replace(0, 1)
        df[feat] = (df[feat] - month_mean) / month_std

    return df.drop(columns=["_month"])


def _add_temporal_lag_features(df: pd.DataFrame, lag_features: list[str], lags: list[int] = [1], rolling_windows: list[int] = [7]) -> pd.DataFrame:
    """
    Add temporal lag and rolling mean features for each node.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with TimeStamp and Compno_x columns
    lag_features : list[str]
        List of feature columns to create lags for
    lags : list[int]
        List of lag periods (default [1] = previous day)
    rolling_windows : list[int]
        List of rolling window sizes for mean (default [7] = 7-day average)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added lag and rolling features
    """
    df = df.copy()
    df = df.sort_values(["Compno_x", "TimeStamp"])
    
    new_features = []
    
    for feat in lag_features:
        # Lag features (previous day values)
        for lag in lags:
            lag_col = f"{feat}_lag{lag}"
            df[lag_col] = df.groupby("Compno_x")[feat].shift(lag)
            new_features.append(lag_col)
        
        # Rolling mean features (e.g., 7-day average)
        for window in rolling_windows:
            roll_col = f"{feat}_roll{window}"
            df[roll_col] = df.groupby("Compno_x")[feat].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            new_features.append(roll_col)
        
        # Difference from previous day (captures trend/change)
        diff_col = f"{feat}_diff"
        df[diff_col] = df.groupby("Compno_x")[feat].diff()
        new_features.append(diff_col)
    
    # Fill NaN values created by lag/diff with 0 (first timestamps)
    for col in new_features:
        df[col] = df[col].fillna(0)
    
    print(f"Added {len(new_features)} temporal lag features:")
    for feat in new_features[:5]:
        print(f"  - {feat}")
    if len(new_features) > 5:
        print(f"  ... and {len(new_features) - 5} more")
    
    return df


def create_temporal_dataset(
    final_df: pd.DataFrame,
    seasonal_normalization: bool = True,
    add_lag_features: bool = True,
    n_classes: int = 5,
    test_holdout_ratio: float = 0.2,
    stratified_split: bool = False,
    forecast_horizon: int = 1,
    target_type: str = "feature_mean",
    target_window: int = 3,
    base_path=None,
    cv_fold: int | None = None,
    n_outer_folds: int = 5,
) -> DynamicGraphTemporalSignalBatch:
    """
    Creates a temporal dataset from the final processed dataframe.

    **This builds a one-step-ahead FORECASTING dataset, not a nowcasting
    one.** Node features `x` at snapshot t are the station's observed
    weather context for month t (current-month features plus lag / rolling
    / diff history and Fourier seasonal encoding). The label `y` at
    snapshot t is the discretised quality category of month **t+1** — each
    station's continuous quality signal is shifted one month forward before
    discretisation. The most recent timestamp in the input data has no
    t+1 and is therefore dropped entirely (142 raw timestamps -> up to 141
    usable snapshots). This mirrors the `Shifted_Target` construction in
    `baselines/baseline_models.py`, so the GNN and baseline pipelines solve
    the same forecasting task and are directly comparable.

    Preprocessing is **leak-free**: the temporal partitioning that will
    eventually separate trainval from the held-out test set is reproduced
    up front to produce a boolean trainval mask over `final_df` rows. That
    mask is then used to fit every downstream parameter (seasonal monthly
    stats, MinMax scaler statistics, and tercile discretisation edges on
    the FORECAST target) on trainval rows only, before applying them to the
    full dataframe. The returned dataset object is unchanged in structure —
    the split logic in the training pipeline still operates over all
    snapshots — but the values inside it have been computed without
    allowing test statistics to influence how training samples are
    transformed or classified.

    Parameters
    ----------
    final_df : pd.DataFrame
        The preprocessed dataframe with node features and edges.
    seasonal_normalization : bool, default True
        If True, normalize features relative to their monthly statistics
        (removes seasonal drift). If False, use global MinMaxScaler.
    add_lag_features : bool, default True
        If True, add lag-1, 7-day rolling mean, and diff features for
        key weather variables to capture temporal autocorrelation.
    n_classes : int, default 5
        Number of classes for target discretization.
    test_holdout_ratio : float, default 0.2
        Fraction of timestamps reserved for the final held-out test set;
        used here only to identify which rows belong to trainval so that
        all fit-time preprocessing parameters are learned on trainval rows.
        Must match the value passed to the outer split later in the
        training pipeline for the mask to align correctly.
    stratified_split : bool, default False
        Match the caller's outer split strategy: True → interleaved
        every-k-th-timestamp split; False → chronological first-N split.
    """
    print("Creating temporal dataset...")

    # Add temporal lag features BEFORE normalization
    if add_lag_features:
        # Key features that benefit from lag information
        lag_feature_cols = [
            "AverageMaximumTemperature_x",
            "AverageTotalRadiation_x",  # GHI is highly autocorrelated
            "TotalRainfall_x",
        ]
        # Only add lags for features that exist
        lag_feature_cols = [f for f in lag_feature_cols if f in final_df.columns]
        if lag_feature_cols:
            final_df = _add_temporal_lag_features(final_df, lag_feature_cols)

    # Label encode Compno columns to be 0-indexed
    le = LabelEncoder()
    final_df["Compno_x_encoded"] = le.fit_transform(final_df["Compno_x"])
    final_df["Compno_y_encoded"] = le.transform(final_df["Compno_y"])
    print("Label encoding complete.")

    # Identify features to normalize
    features_to_scale = [
        col
        for col in final_df.columns
        if "_x" in col and "Compno" not in col and "Target" not in col and "TimeStamp" not in col
    ]
    print("\n" + "=" * 50)
    print(" " * 18 + "DATASET FEATURES")
    print("=" * 50)
    print("Features being scaled:")
    for feature in features_to_scale:
        print(f"- {feature}")
    print("=" * 50)

    # -----------------------------------------------------------------
    # Trainval mask — reproduce the outer temporal split up-front so that
    # every fit-time operation below sees only trainval rows. This is the
    # single structural change that makes the leak fix possible without
    # restructuring the rest of the pipeline.
    # -----------------------------------------------------------------
    sorted_ts = sorted(final_df["TimeStamp"].unique())
    n_total_ts = len(sorted_ts)
    _cv_test_ts: set | None = None
    n_test_ts = max(1, int(round(n_total_ts * test_holdout_ratio)))
    if cv_fold is not None:
        # Rotating outer fold, assigned round-robin WITHIN each calendar month.
        #
        # A naive `position % n_folds` rule looks equivalent but is not: calendar
        # months repeat on a 12-cycle, so whenever gcd(n_folds, 12) != 1 every
        # occurrence of a given month lands in the same fold. All of that month's
        # rows then sit on one side of the split, the seasonal reference period
        # for the other side is empty, and SPI standardisation returns all-NaN.
        # (n_folds=5 survives only because 5 and 12 are coprime.)
        #
        # Cycling within each calendar month instead guarantees every month is
        # spread evenly across folds for ANY fold count, and still gives each
        # month exactly one turn in the test set across the full sweep.
        _fold_dt = _parse_timestamps(pd.Series(sorted_ts))
        _seen: dict[int, int] = {}
        _fold_of_position: list[int] = []
        for _m in _fold_dt.dt.month:
            _k = _seen.get(int(_m), 0)
            _seen[int(_m)] = _k + 1
            _fold_of_position.append(_k % int(n_outer_folds))
        test_ts_positions = {i for i, f in enumerate(_fold_of_position)
                             if f == int(cv_fold)}
        trainval_ts = [ts for i, ts in enumerate(sorted_ts) if i not in test_ts_positions]
        # Keyed by timestamp VALUE so it survives the successor-drop filtering
        # applied further down, which removes rows and shifts positions.
        _cv_test_ts = {ts for i, ts in enumerate(sorted_ts) if i in test_ts_positions}
        print(f"Outer CV fold {cv_fold}/{n_outer_folds}: "
              f"{len(test_ts_positions)} test months, {len(trainval_ts)} trainval months.")
    elif stratified_split:
        # Every-k-th-timestamp scheme — mirrors interleaved_temporal_split.
        step = n_total_ts / n_test_ts
        test_ts_positions = {min(int(i * step), n_total_ts - 1) for i in range(n_test_ts)}
        trainval_ts = [ts for i, ts in enumerate(sorted_ts) if i not in test_ts_positions]
    else:
        # Chronological first-N scheme — mirrors temporal_signal_split.
        trainval_ts = sorted_ts[: n_total_ts - n_test_ts]
    trainval_mask = final_df["TimeStamp"].isin(trainval_ts)
    print(f"Trainval mask: {int(trainval_mask.sum())}/{len(final_df)} rows "
          f"({100.0 * trainval_mask.mean():.1f}%) — used for leak-free fitting below.")

    # Apply normalization based on configuration (all fits on trainval only).
    if seasonal_normalization:
        print("Applying SEASONAL normalization (per-month z-scores)...")
        print("  Monthly stats fit on TRAINVAL rows only, applied to full frame.")
        seasonal_stats = _compute_seasonal_stats(
            final_df[trainval_mask], features_to_scale
        )
        final_df = _apply_seasonal_normalization(final_df, features_to_scale, seasonal_stats)
        # After seasonal z-scoring, rescale to 0-1 for consistency.
        scaler = MinMaxScaler()
        scaler.fit(final_df.loc[trainval_mask, features_to_scale])
        final_df[features_to_scale] = scaler.transform(final_df[features_to_scale])
        print("Seasonal normalization + MinMax rescaling complete (trainval-fit).")
    else:
        print("Applying GLOBAL normalization (MinMaxScaler across all timestamps)...")
        print("  Warning: Seasonal drift patterns will be preserved in the features.")
        print("  Scaler fit on TRAINVAL rows only, applied to full frame.")
        scaler = MinMaxScaler()
        scaler.fit(final_df.loc[trainval_mask, features_to_scale])
        final_df[features_to_scale] = scaler.transform(final_df[features_to_scale])
        print("Feature scaling complete (trainval-fit).")

    # Recompute Target_x_continuous as the mean of the NOW-normalized feature
    # columns. This is the *same-month* composite quality signal — it is not
    # the forecasting label itself, only the raw material the label is built
    # from (see the forward-shift step immediately below).
    _tt = str(target_type).lower()
    if _tt in ("spi", "spei"):
        # Standardised drought index computed from the RAW station record, so
        # the label is derived independently of the model's input features.
        # This removes the circularity of the feature-mean target, where the
        # label was a deterministic function of the inputs.
        from masters_research.pipeline.drought_index import build_index_lookup

        _root = Path(base_path) if base_path is not None else Path(__file__).resolve().parents[3]
        _ref_ts = pd.to_datetime(pd.Series(trainval_ts))   # leak-free reference period
        idx_df = build_index_lookup(_root, window=int(target_window),
                                    use_pet=(_tt == "spei"),
                                    reference_timestamps=_ref_ts)
        print(f"{_tt.upper()}-{target_window} target: {len(idx_df)} station-months computed "
              f"from raw records (reference period = trainval only).")

        _dt = _parse_timestamps(final_df["TimeStamp"])
        merged = pd.DataFrame({"Compno": final_df["Compno_x"].values, "TimeStamp": _dt.values})
        merged = merged.merge(idx_df, on=["Compno", "TimeStamp"], how="left")
        final_df["Target_x_continuous"] = merged["index_value"].values

        # Station-months with no index value (index warm-up, or a station
        # missing from the raw record that month) are KEPT, not dropped. The
        # processed frame supplies their input features either way; only the
        # label is unknown. Dropping them would shrink the node set from one
        # month to the next, which breaks both the fixed spatial graph and the
        # GRU hidden state carried across snapshots. They are instead marked
        # unlabelled below and masked out of the loss and the metrics.
        n_missing = int(final_df["Target_x_continuous"].isna().sum())
        print(f"  {n_missing} of {len(final_df)} rows have no index value "
              f"(warm-up months / stations absent from the raw record); "
              f"these are kept as unlabelled and masked out of the loss.")
    else:
        final_df["Target_x_continuous"] = final_df[features_to_scale].mean(axis=1)
        print("Target_x_continuous recomputed from normalized features (same-month).")

    # -----------------------------------------------------------------
    # Forecasting shift — this study predicts NEXT month's quality category
    # from the CURRENT month's observed context, not the current month's own
    # category. To build that label, each station's same-month continuous
    # target is shifted one month forward in time: the label attached to
    # month t is the (not-yet-discretised) quality signal observed at t+1.
    #
    # final_df is an edge-list (one row per station per neighbour-edge per
    # timestamp), so the shift is computed on a deduplicated one-row-per
    # -station-month table and merged back.
    #
    # The pairing is done by EXPLICIT CALENDAR MATCHING, not by positional
    # shift(-1). A positional shift takes "the next row in this station's
    # sorted sequence", which silently pairs month t with a non-adjacent
    # month wherever the record has a gap — producing a mislabelled example
    # that looks like a valid t+1 target. Matching on
    # `TimeStamp + DateOffset(months=1)` instead means a month with no true
    # successor yields NaN and is dropped, and it mirrors the equivalent
    # construction in `baselines/baseline_models.py` exactly, keeping the two
    # pipelines' labels definitionally identical.
    # -----------------------------------------------------------------
    if forecast_horizon <= 0:
        # Nowcasting mode (horizon 0): the label is the CURRENT month's own
        # category. Retained as an explicit ablation arm — note that under
        # this setting the target is a near-deterministic function of the
        # model's own input features, so results are not evidence of
        # forecasting skill. See the nowcast-vs-forecast discussion.
        print("Nowcasting mode (forecast_horizon=0): label is the CURRENT month.")
        final_df["Target_x_continuous_forecast"] = final_df["Target_x_continuous"]
    else:
        print(f"Building {forecast_horizon}-month-ahead forecasting target "
              f"(t -> t+{forecast_horizon}, calendar-matched)...")
        node_month_targets = (
            final_df[["Compno_x", "TimeStamp", "Target_x_continuous"]]
            .drop_duplicates(subset=["Compno_x", "TimeStamp"])
            .copy()
        )
        # Normalise to real datetimes so DateOffset arithmetic is meaningful
        # for both the ISO-string and Unix-epoch storage formats.
        node_month_targets["_dt"] = _parse_timestamps(node_month_targets["TimeStamp"])

        # Right-hand side: the successor month's target, keyed by the month it
        # is the successor OF (shifted back so it joins onto month t).
        successor = node_month_targets[["Compno_x", "_dt", "Target_x_continuous"]].copy()
        successor["_dt"] = successor["_dt"] - pd.DateOffset(months=forecast_horizon)
        successor = successor.rename(
            columns={"Target_x_continuous": "Target_x_continuous_forecast"}
        )

        node_month_targets = node_month_targets.merge(
            successor, on=["Compno_x", "_dt"], how="left"
        )
        final_df = final_df.merge(
            node_month_targets[["Compno_x", "TimeStamp", "Target_x_continuous_forecast"]],
            on=["Compno_x", "TimeStamp"], how="left",
        )

        # Any station-month whose true calendar successor is absent (the final
        # month of the record, or a month sitting just before a gap) has no
        # valid forecasting label and is dropped — mirroring the dropna applied
        # to `Shifted_Target` in the baseline pipeline.
        # Two different reasons the forecast target can be missing, and they
        # need opposite treatment:
        #   (a) the successor MONTH is absent from the record entirely (the
        #       last month, or a month just before a gap) — no station has a
        #       label, so the whole snapshot is dropped;
        #   (b) the successor month exists but that station has no index value
        #       — the snapshot stays, and this node is simply unlabelled.
        n_before = final_df["TimeStamp"].nunique()
        rows_before = len(final_df)
        _all_dt = set(_parse_timestamps(final_df["TimeStamp"]))
        _row_dt = _parse_timestamps(final_df["TimeStamp"])
        has_successor = (_row_dt + pd.DateOffset(months=forecast_horizon)).isin(_all_dt)
        final_df = final_df.loc[has_successor].reset_index(drop=True)
        n_after = final_df["TimeStamp"].nunique()
        print(f"  Calendar-matched t+{forecast_horizon}: {n_before} -> {n_after} timestamps, "
              f"{rows_before} -> {len(final_df)} rows "
              f"({rows_before - len(final_df)} dropped for missing successor month).")

    # Re-derive the trainval mask against the filtered frame. trainval_ts is
    # a list of timestamp *values* (unaffected by row filtering), so this
    # simply re-applies the same partition boundary already established
    # above, now aligned to final_df's post-drop index.
    trainval_mask = final_df["TimeStamp"].isin(trainval_ts)

    # -----------------------------------------------------------------
    # Discretize the FORECASTING target — bin edges fit on TRAINVAL rows
    # only, then applied to the full frame. Test-set samples that fall
    # outside the trainval range are clipped to the nearest tier so no NaN
    # labels propagate downstream. Test-set class supports may therefore be
    # uneven — this honestly reflects distribution shift rather than hiding
    # it by construction.
    # -----------------------------------------------------------------
    print(f"Discretizing into {n_classes} classes (bin edges fit on TRAINVAL forecast targets only)...")
    trainval_targets = final_df.loc[trainval_mask, "Target_x_continuous_forecast"]

    try:
        _, bin_edges = pd.qcut(
            trainval_targets, q=n_classes, labels=False,
            duplicates="drop", retbins=True,
        )
    except ValueError:
        bin_edges = None

    if bin_edges is None or len(bin_edges) - 1 < n_classes:
        print(
            f"  WARNING: pd.qcut on trainval produced fewer than {n_classes} bins. "
            f"Falling back to equal-width pd.cut over the trainval range."
        )
        _, bin_edges = pd.cut(trainval_targets, bins=n_classes, labels=False, retbins=True)

    # Pad outer edges so any test-only extreme values still map inside a bin.
    eps = 1e-9
    bin_edges = np.asarray(bin_edges, dtype=float).copy()
    bin_edges[0] -= eps
    bin_edges[-1] += eps

    binned = pd.cut(
        final_df["Target_x_continuous_forecast"], bins=bin_edges,
        labels=False, include_lowest=True,
    )
    # Any residual out-of-range test sample (should be near-impossible after
    # the epsilon padding) is clipped to the nearest tier by direction.
    below = final_df["Target_x_continuous_forecast"] < bin_edges[0]
    above = final_df["Target_x_continuous_forecast"] > bin_edges[-1]
    binned = binned.astype("float")
    binned.loc[below] = 0
    binned.loc[above] = n_classes - 1
    # Target_x is the column name every downstream consumer (snapshot
    # builder, training pipeline, baselines comparison) expects. Its VALUE
    # now encodes next-month's category, not the current month's — this is
    # the one-line hinge that converts the pipeline from nowcasting to
    # one-step-ahead forecasting.
    # Rows with no index value carry no label. -1 is the sentinel the training
    # loop masks on (and torch's standard ignore_index), keeping the node in
    # the graph as a feature provider for its neighbours while contributing
    # nothing to the loss or the reported metrics.
    binned.loc[final_df["Target_x_continuous_forecast"].isna()] = -1
    final_df["Target_x"] = binned.astype(int)

    # Report the resulting distribution per partition so the trainval/test
    # split is honest and visible in the training log.
    print("Target discretization complete. Distribution:")
    for label, mask in [("Trainval", trainval_mask), ("Test    ", ~trainval_mask)]:
        _lab = final_df.loc[mask, "Target_x"]
        dist = _lab[_lab >= 0].value_counts().sort_index()
        total = int(dist.sum())
        if total == 0:
            continue
        print(f"  [{label}]")
        for cls, cnt in dist.items():
            print(f"    Class {int(cls)}: {cnt} samples ({100.0 * cnt / total:.1f}%)")

    # Timestamps
    timestamps = final_df["TimeStamp"].unique()

    # Initialize lists to store data for each timestamp
    edge_indices_list = []
    edge_weights_list = []
    node_features_list = []
    targets_list = []
    all_batches = []
    timestamps_list = []

    print("\n" + "=" * 50)
    print(" " * 18 + "DATASET COUNTS")
    print("=" * 50)
    print(f"Number of unique timestamps: {len(timestamps)}")
    print(f"Total number of nodes: {len(final_df['Compno_x_encoded'].unique())}")
    print(f"Total number of edges (time-stamped): {len(final_df)}")
    print("=" * 50)

    for timestamp in timestamps:
        df_timestamp = final_df[final_df["TimeStamp"] == timestamp]

        # Convert timestamp string to numerical (Unix timestamp)
        if isinstance(timestamp, str):
            try:
                dt_object = datetime.strptime(timestamp, "%Y-%m-%d")
                timestamp_numeric = dt_object.timestamp()
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {timestamp}")
        else:
            timestamp_numeric = float(timestamp)

        # Dynamically get all feature columns (includes base + lag features)
        # Base features: temperature, humidity, rainfall, radiation, wind, evaporation
        # Lag features: *_lag1, *_roll7, *_diff (if add_lag_features=True)
        base_feature_cols = [
            "AverageMaximumTemperature_x",
            "AverageMinimunTemperature_x",
            "AverageMaximumRelativeHumidity_x",
            "AverageMinimumRelativeHumidity_x",
            "TotalRainfall_x",
            "AverageTotalRadiation_x",
            "AverageWindSpeed_x",
            "AverageTotalRelativeEvaporation_x",
        ]
        
        # Find any lag/rolling/diff features that exist in the dataframe
        lag_feature_cols = [
            col for col in df_timestamp.columns
            if "_lag" in col or "_roll" in col or "_diff" in col
        ]
        
        all_feature_cols = base_feature_cols + lag_feature_cols

        # Node features - deduplicate by Compno_x_encoded ONLY, then select features.
        # Sorting by the LabelEncoder-assigned id (0..N-1) — rather than the raw
        # ComputerNo station id — is what makes row i of `x` correspond to node
        # index i in `edge_indices` below; the raw station ids are arbitrary
        # hardware numbers, not contiguous small integers.
        node_features = (
            df_timestamp[["Compno_x_encoded"] + all_feature_cols]
            .drop_duplicates(subset=["Compno_x_encoded"])
            .sort_values(by="Compno_x_encoded")
        )
        x = torch.tensor(
            node_features[all_feature_cols].values,
            dtype=torch.float,
        )

        # Multi-harmonic Fourier encoding for seasonality
        # Multiple harmonics help the model capture different seasonal patterns:
        #   - 1st harmonic (annual): main seasonal cycle
        #   - 2nd harmonic (semi-annual): wet/dry seasons, equinox patterns  
        #   - Day-of-year: finer granularity than monthly
        # This creates a richer representation that prevents the model from 
        # getting stuck in local minima that only fit one season.
        # Use the format-aware parser: a bare pd.Timestamp() on a Unix-epoch
        # float would be read as nanoseconds and collapse every snapshot onto
        # 1970-01-01, making these encodings constant and information-free.
        ts_dt = (timestamp if isinstance(timestamp, pd.Timestamp)
                 else _parse_timestamps([timestamp]).iloc[0])

        # Day of year (1-366) for finer temporal resolution
        day_of_year = ts_dt.timetuple().tm_yday
        year_progress = day_of_year / 365.25  # Normalized to [0, 1]
        
        # Multi-harmonic encoding (annual + semi-annual cycles)
        # k=1: annual cycle (1 full wave per year)
        # k=2: semi-annual cycle (2 waves per year - captures wet/dry, equinox)
        fourier_features = []
        for k in [1, 2]:  # harmonics
            angle = 2.0 * np.pi * k * year_progress
            fourier_features.extend([float(np.sin(angle)), float(np.cos(angle))])
        
        # Stack: [sin_1, cos_1, sin_2, cos_2] = 4 features
        temporal_enc = torch.tensor(
            [fourier_features] * x.size(0),
            dtype=torch.float,
        )
        x = torch.cat([x, temporal_enc], dim=1)   # [num_nodes, base_features + lag_features + 4]

        # Edge indices must address rows of `x`, which holds only the stations
        # reporting THIS month. `Compno_x_encoded` is a global 0..N-1 code fit
        # once over every station, so it is only a valid row index when every
        # station is present. Targets such as SPI are defined solely where the
        # raw record exists, leaving months with a subset of stations — so the
        # global codes are remapped to snapshot-local positions here.
        present = node_features["Compno_x_encoded"].to_numpy()
        local_of = {int(g): i for i, g in enumerate(present)}

        src_global = df_timestamp["Compno_x_encoded"].to_numpy()
        dst_global = df_timestamp["Compno_y_encoded"].to_numpy()
        # An edge survives only if BOTH endpoints reported this month; a
        # neighbour that is absent has no feature row to aggregate from.
        keep = np.isin(src_global, present) & np.isin(dst_global, present)

        edge_indices = torch.tensor(
            np.stack([
                np.array([local_of[int(v)] for v in src_global[keep]], dtype=np.int64),
                np.array([local_of[int(v)] for v in dst_global[keep]], dtype=np.int64),
            ]),
            dtype=torch.long,
        )

        # Edge weights: INVERSE distance so closer stations have HIGHER influence
        # Using exp(-distance) for smooth decay; epsilon prevents division by zero
        distances = df_timestamp["distance_km"].values[keep]
        edge_weights = torch.tensor(np.exp(distances * -1.0), dtype=torch.float)

        # Targets - must match node_features extraction (one per node, same order)
        targets_df = (
            df_timestamp[["Compno_x_encoded", "Target_x"]]
            .drop_duplicates(subset=["Compno_x_encoded"])
            .sort_values(by="Compno_x_encoded")
        )
        targets_np = np.array(targets_df["Target_x"].values, dtype=np.float64)
        targets_tensor = torch.tensor(targets_np, dtype=torch.float)
        
        # Sanity check: node features and targets must have same count
        assert x.size(0) == targets_tensor.size(0), (
            f"Node/target mismatch at timestamp {timestamp}: "
            f"{x.size(0)} nodes vs {targets_tensor.size(0)} targets"
        )

        # Generate batches (assuming each timestamp is a separate batch)
        batch = torch.arange(x.size(0), dtype=torch.long)

        snapshot_timestamp = torch.FloatTensor([timestamp_numeric])

        # Append to lists
        node_features_list.append(x)
        edge_indices_list.append(edge_indices)
        edge_weights_list.append(edge_weights)
        targets_list.append(targets_tensor)
        all_batches.append(batch)
        timestamps_list.append(snapshot_timestamp)

    # Ensure the length consistency across all lists
    assert (
        len(edge_indices_list)
        == len(edge_weights_list)
        == len(node_features_list)
        == len(targets_list)
        == len(all_batches)
        == len(timestamps_list)
    ), "Temporal dimension inconsistency."

    # Per-snapshot test flag, aligned with the snapshot lists above. Carrying it
    # on the dataset lets interleaved_temporal_split reproduce the split exactly
    # instead of recomputing it — the timestamps tensor is float32 and cannot
    # round-trip a Unix epoch, so re-deriving months from it would be unsafe.
    _cv_is_test = ([bool(ts in _cv_test_ts) for ts in timestamps]
                   if _cv_test_ts is not None else None)

    print("Temporal dataset creation complete.")

    # Reconstruct the exact per-snapshot feature ordering so downstream explainability
    # tools (GNNExplainer / Integrated Gradients) can label axes with real names
    # instead of numeric indices. Mirrors the assembly inside the snapshot loop above:
    #   [base_feature_cols] + [detected lag/roll/diff cols] + Fourier(k=1,2) = [sin_1, cos_1, sin_2, cos_2]
    _base_feature_cols = [
        "AverageMaximumTemperature_x",
        "AverageMinimunTemperature_x",
        "AverageMaximumRelativeHumidity_x",
        "AverageMinimumRelativeHumidity_x",
        "TotalRainfall_x",
        "AverageTotalRadiation_x",
        "AverageWindSpeed_x",
        "AverageTotalRelativeEvaporation_x",
    ]
    _lag_feature_cols = [
        col for col in final_df.columns
        if ("_lag" in col or "_roll" in col or "_diff" in col)
    ]
    _fourier_feature_cols = ["fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2"]
    feature_names = _base_feature_cols + _lag_feature_cols + _fourier_feature_cols

    dataset = DynamicGraphTemporalSignalBatch(
        edge_indices=edge_indices_list,
        edge_weights=edge_weights_list,
        features=node_features_list,
        targets=targets_list,
        batches=all_batches,
        timestamps=timestamps_list,
    )
    dataset.feature_names = feature_names
    dataset.cv_is_test = _cv_is_test
    return dataset


def stratified_temporal_signal_split(dataset: DynamicGraphTemporalSignalBatch, train_ratio: float = 0.8, num_classes: int | None = None):
    """
    Split snapshots into train/val while preserving class proportions.

    Uses interleaved sampling within class-based stratification to maintain both
    temporal patterns and class balance across splits.

    Algorithm:
    1. Compute signature vector for each snapshot (class counts as proportion)
    2. Use k-means to cluster snapshots by class signature similarity
    3. Within each cluster, apply interleaved sampling to select val snapshots
    4. This ensures val set mirrors train set class distribution

    Returns (train_dataset, val_dataset).
    """
    total = dataset.snapshot_count
    val_count = max(1, int(round(total * (1.0 - float(train_ratio)))))

    # Compute per-snapshot class counts
    snapshot_class_counts = []
    for i in range(total):
        snap = dataset[i]
        lbls = snap.y.cpu().numpy().astype(int)
        lbls = lbls[lbls >= 0]      # unlabelled nodes carry no class signature
        counts = np.bincount(lbls, minlength=num_classes if num_classes else 0)
        snapshot_class_counts.append(counts)

    # Infer number of classes if not provided
    if num_classes is None:
        num_classes = max(len(c) for c in snapshot_class_counts) if snapshot_class_counts else 1

    # Pad counts to num_classes if needed
    snapshot_class_counts = [
        np.pad(c, (0, max(0, num_classes - len(c)))) for c in snapshot_class_counts
    ]

    # Convert to proportions (signature vectors)
    signatures = []
    for counts in snapshot_class_counts:
        total_nodes = counts.sum()
        if total_nodes > 0:
            signatures.append(counts / total_nodes)
        else:
            signatures.append(counts.astype(float))
    signatures = np.array(signatures)  # (num_snapshots, num_classes)

    # Compute overall target class distribution from all snapshots
    all_counts = np.sum(snapshot_class_counts, axis=0)
    target_dist = all_counts / all_counts.sum() if all_counts.sum() > 0 else np.ones(num_classes) / num_classes

    # Strategy: select val snapshots that together approximate the target distribution
    # while maintaining temporal spread
    
    # Step 1: Interleaved base selection (ensures temporal coverage)
    step = total / val_count
    base_val_idxs = sorted({min(int(i * step), total - 1) for i in range(val_count)})

    # Step 2: Compute class distribution of base selection
    def compute_dist(idxs):
        counts = np.sum([snapshot_class_counts[i] for i in idxs], axis=0)
        return counts / counts.sum() if counts.sum() > 0 else np.ones(num_classes) / num_classes

    # Step 3: Iteratively swap to improve class balance (greedy refinement)
    val_idxs = list(base_val_idxs)
    remaining = [i for i in range(total) if i not in set(val_idxs)]
    
    # JS divergence helper
    def js_divergence(p, q):
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        m = 0.5 * (p + q)
        return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

    max_swaps = min(50, total)  # limit iterations
    for _ in range(max_swaps):
        current_dist = compute_dist(val_idxs)
        current_js = js_divergence(current_dist, target_dist)
        
        # Try swapping each val snapshot with each remaining snapshot
        best_swap = None
        best_js = current_js
        
        for vi, val_i in enumerate(val_idxs):
            for rem_i in remaining:
                # Candidate swap
                candidate = val_idxs.copy()
                candidate[vi] = rem_i
                cand_dist = compute_dist(candidate)
                cand_js = js_divergence(cand_dist, target_dist)
                
                if cand_js < best_js - 1e-6:  # must improve significantly
                    best_js = cand_js
                    best_swap = (vi, val_i, rem_i)
        
        if best_swap is None:
            break  # no improvement possible
        
        vi, val_i, rem_i = best_swap
        val_idxs[vi] = rem_i
        remaining.remove(rem_i)
        remaining.append(val_i)

    val_set = set(val_idxs)
    train_idxs = [i for i in range(total) if i not in val_set]
    val_idxs = sorted(val_idxs)  # maintain temporal order

    # Construct new datasets by indexing
    def subset(idxs):
        return DynamicGraphTemporalSignalBatch(
            edge_indices=[dataset.edge_indices[i] for i in idxs],
            edge_weights=[dataset.edge_weights[i] for i in idxs],
            features=[dataset.features[i] for i in idxs],
            targets=[dataset.targets[i] for i in idxs],
            batches=[dataset.batches[i] for i in idxs],
            timestamps=[dataset.timestamps[i] for i in idxs],
        )

    return subset(train_idxs), subset(val_idxs)


def interleaved_temporal_split(
    dataset: DynamicGraphTemporalSignalBatch, train_ratio: float = 0.8,
    cv_fold: int | None = None, n_outer_folds: int = 5,
):
    """
    Systematic (interleaved) outer split that preserves the full temporal class distribution.

    Every ~(1/test_ratio)-th snapshot is assigned to the test set so that every season
    is proportionally represented in both partitions.  Temporal ordering within each
    partition is preserved.

    This prevents the seasonal label drift that occurs when the last N% of chronological
    snapshots are used as the test set.
    """
    total = dataset.snapshot_count
    if cv_fold is not None:
        # Read the assignment recorded when the dataset was built, rather than
        # recomputing it. The two must agree exactly -- if they ever diverged,
        # the scalers and bin edges would be fit on months that end up in this
        # fold's test set, silently reintroducing the leak this design removes.
        flags = getattr(dataset, "cv_is_test", None)
        if flags is None:
            raise ValueError(
                "interleaved_temporal_split(cv_fold=...) needs a dataset built by "
                "create_temporal_dataset(cv_fold=...); this one carries no fold "
                "assignment, so the split could not be guaranteed leak-free."
            )
        if len(flags) != total:
            raise ValueError(
                f"Fold assignment covers {len(flags)} snapshots but the dataset has "
                f"{total}; refusing to split on a stale assignment."
            )
        test_idxs = [i for i, is_test in enumerate(flags) if is_test]
        test_set = set(test_idxs)
        train_idxs = [i for i in range(total) if i not in test_set]

        def _subset_cv(idxs):
            return DynamicGraphTemporalSignalBatch(
                edge_indices=[dataset.edge_indices[i] for i in idxs],
                edge_weights=[dataset.edge_weights[i] for i in idxs],
                features=[dataset.features[i] for i in idxs],
                targets=[dataset.targets[i] for i in idxs],
                batches=[dataset.batches[i] for i in idxs],
                timestamps=[dataset.timestamps[i] for i in idxs],
            )
        return _subset_cv(train_idxs), _subset_cv(test_idxs)

    test_count = max(1, int(round(total * (1.0 - train_ratio))))
    step = total / test_count  # float step so positions spread evenly
    test_idxs: list[int] = sorted({min(int(i * step), total - 1) for i in range(test_count)})
    # If rounding / set-deduplication left us short, fill from the middle of the series
    if len(test_idxs) < test_count:
        remaining = [i for i in range(total) if i not in set(test_idxs)]
        mid = len(remaining) // 2
        fill = remaining[mid: mid + (test_count - len(test_idxs))]
        test_idxs = sorted(test_idxs + fill)

    test_set   = set(test_idxs)
    train_idxs = [i for i in range(total) if i not in test_set]

    def _subset(idxs):
        return DynamicGraphTemporalSignalBatch(
            edge_indices=[dataset.edge_indices[i] for i in idxs],
            edge_weights=[dataset.edge_weights[i] for i in idxs],
            features=[dataset.features[i] for i in idxs],
            targets=[dataset.targets[i] for i in idxs],
            batches=[dataset.batches[i] for i in idxs],
            timestamps=[dataset.timestamps[i] for i in idxs],
        )

    return _subset(train_idxs), _subset(test_idxs)


def walk_forward_cv_split(dataset: DynamicGraphTemporalSignalBatch, n_folds: int = 5):
    """
    Expanding-window walk-forward cross-validation generator.

    Divides the T snapshots into (n_folds + 1) equal segments of size
    seg = T // (n_folds + 1).  For fold k (1-indexed):

        Training window:   snapshots [0,  (k) * seg)
        Validation window: snapshots [(k) * seg,  min((k+1) * seg, T)]

    The last fold's validation window runs to the end of the dataset.

    Yields (train_dataset, val_dataset, fold_number) for each valid fold.
    """
    total = dataset.snapshot_count
    seg   = max(1, total // (n_folds + 1))

    def _subset(idxs):
        return DynamicGraphTemporalSignalBatch(
            edge_indices=[dataset.edge_indices[i] for i in idxs],
            edge_weights=[dataset.edge_weights[i] for i in idxs],
            features=[dataset.features[i] for i in idxs],
            targets=[dataset.targets[i] for i in idxs],
            batches=[dataset.batches[i] for i in idxs],
            timestamps=[dataset.timestamps[i] for i in idxs],
        )

    for k in range(n_folds):
        train_end = seg * (k + 1)
        val_start = train_end
        val_end   = min(val_start + seg, total) if k < n_folds - 1 else total

        if train_end >= total or val_start >= total or val_end <= val_start:
            print(f"  Walk-forward CV: fold {k + 1} skipped (insufficient snapshots).")
            continue

        yield _subset(list(range(train_end))), _subset(list(range(val_start, val_end))), k + 1
