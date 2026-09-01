"""Standardised drought indices (SPI / SPEI) computed from the raw station record.

Motivation
----------
The project's original target was ``tercile(mean of the 21 normalised input
features)``. That construction has three defects that make it indefensible as a
drought label:

1. **Circular** - the target is a deterministic function of the model's own
   inputs, so a no-learning rule (average the inputs, split into terciles)
   scores ~0.98 weighted F1. Models were being scored on arithmetic, not skill.
2. **Physically incoherent** - it averages normalised max-temperature, humidity,
   rainfall, radiation, wind and evaporation. Higher temperature implies *more*
   drought while higher rainfall implies *less*, so the terms partially cancel
   and the composite tracks no real physical quantity.
3. **Arbitrary thresholds** - terciles force a 33/33/33 split, implying a third
   of all months are severe drought.

SPI (McKee et al., 1993; WMO-endorsed) and SPEI (Vicente-Serrano et al., 2010)
fix all three. Both are computed here from the *raw* station files rather than
the processed feature matrix, so the label is derived independently of the model
inputs.

Method
------
1. Water-balance term ``D``:
     * SPI  -> ``D = P``            (precipitation only)
     * SPEI -> ``D = P - PET``      (climatic water balance)
2. Accumulate ``D`` over a ``window``-month rolling sum per station.
3. Standardise within each calendar month, so a January is compared against
   other Januaries rather than against the annual distribution.
4. The result is a z-score: negative = drier than normal, positive = wetter.

Standardisation is **nonparametric** (empirical rank -> normal quantile), after
Farahmand & AghaKouchak (2015). This avoids fitting a gamma / log-logistic
distribution to short records, where fits are unstable, while remaining a
citable published variant.

Leak-free operation
-------------------
Conventionally an index is standardised against the full climatological record.
That would let held-out months influence the reference distribution used to
label training months. ``reference_mask`` restricts the fitted reference to
trainval rows only, matching the leak-free discipline used elsewhere in this
pipeline; the resulting mapping is then applied to every month.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# Column names as they appear in Dataset/OriginalDataset/St_*.csv
_RAW_PRECIP = "Rain"
_RAW_PET = "ET0"


def load_raw_station_records(base_path: Path) -> pd.DataFrame:
    """Read every per-station CSV into one long frame.

    The raw files are semicolon-separated with comma decimals and encode
    missing values as ``~~``; everything is coerced to numeric so those become
    NaN rather than silently poisoning later arithmetic with strings.
    """
    pattern = str(Path(base_path) / "Dataset" / "OriginalDataset" / "St_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No raw station files matched {pattern}")

    frames = [pd.read_csv(f, sep=";", decimal=",") for f in files]
    df = pd.concat(frames, ignore_index=True)

    for col in ("Compno", "Year", "Month", _RAW_PRECIP, _RAW_PET):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Compno", "Year", "Month", _RAW_PRECIP])
    df[["Compno", "Year", "Month"]] = df[["Compno", "Year", "Month"]].astype(int)
    df["TimeStamp"] = pd.to_datetime(
        dict(year=df.Year, month=df.Month, day=1), errors="coerce"
    )
    return df.dropna(subset=["TimeStamp"]).sort_values(["Compno", "TimeStamp"])


def _standardise_by_month(values: pd.Series, months: pd.Series,
                          reference: pd.Series | None) -> pd.Series:
    """Empirical-rank -> normal-quantile transform, applied per calendar month.

    `reference` is the boolean mask of rows allowed to define the reference
    distribution. Values outside it are still transformed, but contribute
    nothing to the mapping.
    """
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if reference is None:
        reference = pd.Series(True, index=values.index)

    for m in range(1, 13):
        in_month = (months == m) & values.notna()
        ref = values[in_month & reference].dropna()
        if len(ref) < 8:           # too short to define a stable reference
            continue
        ref_sorted = np.sort(ref.values)
        target = values[in_month]
        # Fraction of reference values below each observation -> plotting
        # position in (0, 1), then the matching standard-normal quantile.
        pos = np.searchsorted(ref_sorted, target.values, side="right")
        q = (pos + 0.5) / (len(ref_sorted) + 1.0)
        q = np.clip(q, 1e-6, 1 - 1e-6)
        out.loc[target.index] = norm.ppf(q)
    return out


def compute_index(records: pd.DataFrame, window: int = 3, use_pet: bool = False,
                  reference_mask: pd.Series | None = None) -> pd.DataFrame:
    """Compute SPI (``use_pet=False``) or SPEI (``use_pet=True``).

    Returns a frame of ``[Compno, TimeStamp, index_value]`` with NaN rows -
    the first ``window - 1`` months of each station - already dropped.
    """
    df = records.copy()
    if use_pet:
        if _RAW_PET not in df.columns:
            raise ValueError(f"SPEI requires a '{_RAW_PET}' column in the raw records.")
        df = df.dropna(subset=[_RAW_PET])
        df["_D"] = df[_RAW_PRECIP] - df[_RAW_PET]
    else:
        df["_D"] = df[_RAW_PRECIP]

    df = df.sort_values(["Compno", "TimeStamp"])
    df["_acc"] = (df.groupby("Compno")["_D"]
                    .transform(lambda s: s.rolling(window, min_periods=window).sum()))

    ref = None if reference_mask is None else reference_mask.reindex(df.index).fillna(False)
    df["index_value"] = _standardise_by_month(df["_acc"], df["Month"], ref)

    return (df.dropna(subset=["index_value"])
              [["Compno", "TimeStamp", "index_value"]]
              .reset_index(drop=True))


def build_index_lookup(base_path: Path, window: int = 3, use_pet: bool = False,
                       reference_timestamps=None) -> pd.DataFrame:
    """Convenience wrapper: load the raw record and return the index frame.

    `reference_timestamps` (optional) restricts the standardisation reference to
    those months, which is how the caller keeps the label leak-free.
    """
    records = load_raw_station_records(base_path)
    mask = None
    if reference_timestamps is not None:
        ref_ts = pd.to_datetime(pd.Series(list(reference_timestamps)))
        mask = records["TimeStamp"].isin(set(ref_ts))
    return compute_index(records, window=window, use_pet=use_pet, reference_mask=mask)
