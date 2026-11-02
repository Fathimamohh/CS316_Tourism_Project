"""
Shared helpers for the CS316 tourism project.

All three notebooks import this module, so data loading, feature engineering
and evaluation are defined in a single place.

Contents
1. Loading:   parse sheets D5 (domestic) and I6 (inbound) of the Ministry of
              Tourism workbook into a tidy long table.
2. Memory:    downcast dtypes to reduce memory usage.
3. Calendar:  Ramadan / Hajj / pre-Hajj days per month (known in advance).
4. Features:  lag and calendar features that use only earlier months.
5. Metrics:   MAE, RMSE, WAPE, R2, MAPE and forecast accuracy = 100 - WAPE.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Loading the two Excel sheets
# ---------------------------------------------------------------------------

# Month names used in the sheets -> month number (1-12).
MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}

# Sheet code -> tourist type stored in our tidy table.
SHEETS = {"D5": "Domestic", "I6": "Inbound"}

# The two indicators in the sheets -> short column names.
INDICATORS = {"Tourists Number (Overnight Visitors)": "tourists",
              "Tourists Spending (SAR)": "spending_sar"}


def find_dataset(filename="DDS_MT_Master_Sheet_2021_2025.xlsx"):
    """Return the path of the workbook, searching the usual locations.

    Covers running from the project root, from notebooks/, and from /content
    in Google Colab. Raises FileNotFoundError if the file is in none of them.
    """
    candidates = [Path("data") / filename, Path("../data") / filename,
                  Path(filename), Path("/content") / filename,
                  Path("/content/CS316_Tourism_Project/data") / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{filename} not found. Put it in the data/ folder (see data/README.md).")


def load_sheet(path, sheet):
    """Read one sheet (I6 or D5) and return a tidy table.

    The sheet layout is "wide": one column per destination and two rows per
    month (tourist number row + spending row). The year is written only on
    the first row of each year, and the month only on the number row.
    """
    # header=None: read every cell as data, we locate the header ourselves.
    raw = pd.read_excel(path, sheet_name=sheet, header=None)

    # The header row is located by the "Month" label in column C. It holds the
    # Arabic destination names; the English names are one row below and the
    # (merged) province names one row above.
    hdr = raw.index[raw[2].astype(str).str.strip() == "Month"][0]
    dest_cols = list(range(4, 45))           # 41 destination columns (E..AS)
    total_col = 45                           # column AT = TOTAL
    names = raw.loc[hdr + 1, dest_cols].astype(str).str.strip()
    # Province names appear only above the first city of each province
    # (merged cells), so we forward-fill them to the right.
    provinces = raw.loc[hdr - 1, dest_cols].ffill().astype(str).str.strip()

    data = raw.loc[hdr + 2:, [1, 2, 3] + dest_cols + [total_col]].copy()
    data.columns = ["year", "month", "indicator"] + list(names) + ["TOTAL"]
    # Year and month labels appear only once per block, so forward-fill them.
    # This fills layout labels only, never measured values.
    data[["year", "month"]] = data[["year", "month"]].ffill()
    data = data[data["indicator"].isin(INDICATORS)]      # drop blank rows
    data = data[data["year"].astype(str) != "Total"]     # drop yearly totals
    data = data[data["month"].isin(MONTHS)]              # drop "Total" months

    # Wide -> long: one row per (month, destination, indicator).
    long = data.melt(id_vars=["year", "month", "indicator"],
                     var_name="destination", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")  # blank -> NaN
    tidy = (long.pivot_table(index=["year", "month", "destination"],
                             columns="indicator", values="value",
                             aggfunc="first", dropna=False)
                .rename(columns=INDICATORS).reset_index())
    tidy.columns.name = None

    tidy["year"] = tidy["year"].astype(int)
    tidy["month"] = tidy["month"].map(MONTHS).astype(int)
    tidy["date"] = pd.to_datetime(dict(year=tidy["year"], month=tidy["month"], day=1))
    tidy["tourist_type"] = SHEETS[sheet]
    prov_map = dict(zip(names, provinces.str.replace(" Province", "", regex=False)
                                        .str.replace(" province", "", regex=False)))
    prov_map["TOTAL"] = "All"
    tidy["province"] = tidy["destination"].map(prov_map)
    # "Other X" columns are the remaining governorates of a province, not a city.
    tidy["is_other"] = tidy["destination"].str.startswith("Other")
    return tidy[["date", "year", "month", "tourist_type", "province",
                 "destination", "is_other", "tourists", "spending_sar"]]


def load_tourism(path=None):
    """Load D5 and I6 and return (destinations, national totals).

    `dests` has one row per tourist type, destination and month (41 x 2 x 60),
    sorted so each series is in date order; `totals` holds the official TOTAL column.
    """
    path = path or find_dataset()
    both = pd.concat([load_sheet(path, s) for s in SHEETS], ignore_index=True)
    totals = both[both["destination"] == "TOTAL"].reset_index(drop=True)
    dests = both[both["destination"] != "TOTAL"]
    dests = dests.sort_values(["tourist_type", "destination", "date"]).reset_index(drop=True)
    return dests, totals


def check_totals(dests, totals):
    """Largest relative gap between the sum of destinations and the TOTAL column.

    Used as a parsing check: a value close to 0 means no destination value was lost.
    """
    s = dests.groupby(["tourist_type", "date"])[["tourists", "spending_sar"]].sum()
    t = totals.set_index(["tourist_type", "date"])[["tourists", "spending_sar"]]
    return float(((s - t).abs() / t).max().max())


# ---------------------------------------------------------------------------
# 2. Memory optimisation
# ---------------------------------------------------------------------------

def optimize_memory(df):
    """Return a copy that uses less memory.

    - text columns -> 'category' (each label stored once, rows hold integer codes)
    - integers -> smallest integer type that fits
    - floats -> float32 (~7 significant digits), except spending_sar, which is
      kept as float64 because monthly spending reaches billions of SAR.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype) in ("str", "string"):
            out[col] = out[col].astype("category")
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="integer")
        elif pd.api.types.is_float_dtype(out[col]) and col != "spending_sar":
            out[col] = out[col].astype("float32")
    return out


def memory_mb(df):
    """Deep memory usage of a DataFrame in megabytes."""
    return df.memory_usage(deep=True).sum() / 1024 ** 2


# ---------------------------------------------------------------------------
# 3. Islamic calendar features (Ramadan and Hajj move ~11 days earlier/year)
# ---------------------------------------------------------------------------
# Dates follow the Saudi Umm al-Qura calendar (Ramadan = first to last fasting
# day; Hajj = 8-13 Dhu al-Hijjah). They are published years in advance, so
# using them for forecasting does not leak future tourism information.
RAMADAN = {2020: ("2020-04-24", "2020-05-23"), 2021: ("2021-04-13", "2021-05-12"),
           2022: ("2022-04-02", "2022-05-01"), 2023: ("2023-03-23", "2023-04-20"),
           2024: ("2024-03-11", "2024-04-09"), 2025: ("2025-03-01", "2025-03-29")}
HAJJ = {2020: ("2020-07-29", "2020-08-03"), 2021: ("2021-07-18", "2021-07-23"),
        2022: ("2022-07-07", "2022-07-12"), 2023: ("2023-06-26", "2023-07-01"),
        2024: ("2024-06-14", "2024-06-19"), 2025: ("2025-06-04", "2025-06-09")}


# Pre-Hajj window: about one lunar month (Dhu al-Qa'dah) before Hajj, when Umrah
# visits are restricted. We approximate it as the 30 days before Hajj starts.
PRE_HAJJ = {y: (str((pd.Timestamp(a) - pd.Timedelta(days=30)).date()),
                str((pd.Timestamp(a) - pd.Timedelta(days=1)).date()))
            for y, (a, b) in HAJJ.items()}


def days_in_month(periods, month_start):
    """Count the days of the month starting at `month_start` covered by any period.

    `periods` maps year -> (first_day, last_day), inclusive.
    """
    month_days = pd.date_range(month_start, month_start + pd.offsets.MonthEnd(0))
    event_days = set()
    for start, end in periods.values():
        event_days.update(pd.date_range(start, end))
    return sum(d in event_days for d in month_days)


def calendar_table(dates):
    """Return Ramadan, Hajj and pre-Hajj day counts for each distinct month start in `dates`."""
    dates = pd.DatetimeIndex(sorted(set(dates)))
    return pd.DataFrame({"date": dates,
                         "ramadan_days": [days_in_month(RAMADAN, d) for d in dates],
                         "hajj_days": [days_in_month(HAJJ, d) for d in dates],
                         "prehajj_days": [days_in_month(PRE_HAJJ, d) for d in dates]})


# ---------------------------------------------------------------------------
# 4. Forecasting features (one-month-ahead)
# ---------------------------------------------------------------------------

def eligible_series(dests):
    """Return the (tourist_type, destination) pairs observed and > 0 in every month.

    Only these series are modelled, so no target value is ever imputed. Zeros
    are excluded too, because in I6 a 0 can mean "not reported".
    """
    ok = dests.groupby(["tourist_type", "destination"])["tourists"].apply(
        lambda s: s.notna().all() and (s > 0).all())
    return ok[ok].index


def build_features(dests):
    """Create the modelling table for one-month-ahead forecasts.

    Counts are modelled as log(1 + tourists) because they range from hundreds
    to millions and seasonal effects are multiplicative. Every feature for
    month t uses only months t-1 and earlier, or calendar dates known in advance.
    Rows near the start of each series have NaN lags; callers drop them.
    """
    keep = eligible_series(dests)
    df = dests.set_index(["tourist_type", "destination"]).loc[keep].reset_index()
    df = df.sort_values(["tourist_type", "destination", "date"]).reset_index(drop=True)
    df["log_y"] = np.log1p(df["tourists"].astype(float))
    g = df.groupby(["tourist_type", "destination"], observed=True)["log_y"]

    # lag k = log value k months earlier in the same series. Requires the rows to be
    # sorted by date within each series (done above). lag24 is computed but not
    # used in any feature group below.
    for k in [1, 2, 3, 12, 13, 14, 15, 24]:
        df[f"lag{k}"] = g.shift(k)
    # Year-over-year growth of the last three observed months (differences of logs).
    df["yoy1"] = df["lag1"] - df["lag13"]
    df["yoy2"] = df["lag2"] - df["lag14"]
    df["yoy3"] = df["lag3"] - df["lag15"]
    df["yoy_mean3"] = df[["yoy1", "yoy2", "yoy3"]].mean(axis=1)
    # How month t compared with month t-1 one year ago (expected seasonal step).
    df["seasonal_step"] = df["lag12"] - df["lag13"]
    # Mean of the last three observed months (t-1..t-3), on the log scale.
    df["roll3"] = df[["lag1", "lag2", "lag3"]].mean(axis=1)

    # Cyclical month encoding so December and January are close together.
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    cal = calendar_table(pd.date_range("2020-01-01", "2025-12-01", freq="MS"))
    cal_last_year = cal.assign(date=cal["date"] + pd.DateOffset(years=1))
    df = df.merge(cal, on="date", how="left").merge(
        cal_last_year.rename(columns={"ramadan_days": "ramadan_ly", "hajj_days": "hajj_ly",
                                      "prehajj_days": "prehajj_ly"}),
        on="date", how="left")
    # Calendar change versus the same month last year: lag12 already carries last
    # year's calendar, so the model needs to know how this year differs.
    df["ramadan_change"] = df["ramadan_days"] - df["ramadan_ly"]
    df["hajj_change"] = df["hajj_days"] - df["hajj_ly"]
    df["prehajj_change"] = df["prehajj_days"] - df["prehajj_ly"]
    # Religious seasons mainly affect the Makkah and Madinah provinces, so these
    # interaction terms are zero everywhere else.
    holy = df["province"].isin(["Makkah", "Madinah"]).astype(int)
    for c in ["ramadan_change", "hajj_change", "prehajj_change"]:
        df[f"holy_{c}"] = holy * df[c]
    df["is_inbound"] = (df["tourist_type"] == "Inbound").astype(int)
    df["is_other"] = df["is_other"].astype(int)
    df["series"] = df["tourist_type"].astype(str) + " | " + df["destination"].astype(str)
    return df


# Feature groups compared in notebook 03.
FEATURES_BASIC = ["lag1", "lag12", "lag13"]
FEATURES_LAGS = FEATURES_BASIC + ["lag2", "lag3", "lag14", "lag15",
                                  "yoy1", "yoy2", "yoy3", "yoy_mean3",
                                  "seasonal_step", "roll3"]
FEATURES_CAL = ["month_sin", "month_cos", "ramadan_days", "hajj_days", "prehajj_days",
                "ramadan_change", "hajj_change", "prehajj_change"]
FEATURES_HOLY = ["holy_ramadan_change", "holy_hajj_change", "holy_prehajj_change"]
FEATURES_CAT = ["is_inbound", "is_other", "province"]   # province is one-hot encoded
FEATURES_ALL = FEATURES_LAGS + FEATURES_CAL + FEATURES_HOLY + FEATURES_CAT


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred):
    """Regression metrics on the original scale (number of tourists).

    MAE      = mean |error|                     (tourists)
    RMSE     = sqrt(mean error^2)               (tourists; penalises large misses)
    WAPE     = sum |error| / sum actual * 100   (% of all tourists mis-forecast)
    Accuracy = 100 - WAPE                       (not classification accuracy)
    R2       = 1 - SSE / SST
    MAPE     = mean(|error| / actual) * 100     (each row weighted equally)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    wape = np.abs(err).sum() / np.abs(y_true).sum() * 100
    return {"MAE": np.abs(err).mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "WAPE_%": wape,
            "Accuracy_%": 100 - wape,
            "MAPE_%": np.mean(np.abs(err) / y_true) * 100,
            "R2": 1 - (err ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()}
