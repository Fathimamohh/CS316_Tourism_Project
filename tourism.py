"""
tourism.py - shared helper functions for the CS316 tourism project.

All three notebooks import this file so that the data loading, feature
engineering and evaluation code exists in ONE place only (no copy-paste).

Main parts
1. Loading:   read sheets I6 (inbound) and D5 (domestic) from the Ministry of
              Tourism workbook and reshape them into a tidy "long" table.
2. Memory:    shrink the table with smaller data types (rubric: memory optimisation).
3. Calendar:  Ramadan / Hajj days per month (known in advance -> no leakage).
4. Features:  lag features that only use information from earlier months.
5. Metrics:   MAE, RMSE, WAPE, R2 and the "forecast accuracy" = 100% - WAPE.
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
    """Return the path of the Excel file.

    We search a few likely folders so the same notebook works locally
    (run from notebooks/) and in Google Colab (run from /content).
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

    # The row whose 3rd column says "Month" holds the Arabic city names;
    # the row below it holds the English names, the row above the provinces.
    hdr = raw.index[raw[2].astype(str).str.strip() == "Month"][0]
    dest_cols = list(range(4, 45))           # 41 destination columns (E..AS)
    total_col = 45                           # column AT = TOTAL
    names = raw.loc[hdr + 1, dest_cols].astype(str).str.strip()
    # Province names appear only above the first city of each province
    # (merged cells), so we forward-fill them to the right.
    provinces = raw.loc[hdr - 1, dest_cols].ffill().astype(str).str.strip()

    data = raw.loc[hdr + 2:, [1, 2, 3] + dest_cols + [total_col]].copy()
    data.columns = ["year", "month", "indicator"] + list(names) + ["TOTAL"]
    # Forward fill is valid here: it only copies the label written above
    # (layout), it does NOT fill any measured value.
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
    """Load D5 + I6 and return (destinations table, national TOTAL table)."""
    path = path or find_dataset()
    both = pd.concat([load_sheet(path, s) for s in SHEETS], ignore_index=True)
    totals = both[both["destination"] == "TOTAL"].reset_index(drop=True)
    dests = both[both["destination"] != "TOTAL"]
    dests = dests.sort_values(["tourist_type", "destination", "date"]).reset_index(drop=True)
    return dests, totals


def check_totals(dests, totals):
    """Largest relative gap between the sum of destinations and the TOTAL column.

    A value close to 0 proves that our parsing kept every destination value.
    """
    s = dests.groupby(["tourist_type", "date"])[["tourists", "spending_sar"]].sum()
    t = totals.set_index(["tourist_type", "date"])[["tourists", "spending_sar"]]
    return float(((s - t).abs() / t).max().max())


# ---------------------------------------------------------------------------
# 2. Memory optimisation
# ---------------------------------------------------------------------------

def optimize_memory(df):
    """Return a copy that uses less memory.

    - text columns with few distinct values -> 'category' (stored as codes)
    - int64 -> smallest integer type, float64 -> float32
    float32 keeps ~7 significant digits, enough for counts below 16 million.
    Spending (billions of SAR) is kept as float64 to avoid rounding money.
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
    """Number of days of the given events that fall inside the month."""
    month_days = pd.date_range(month_start, month_start + pd.offsets.MonthEnd(0))
    event_days = set()
    for start, end in periods.values():
        event_days.update(pd.date_range(start, end))
    return sum(d in event_days for d in month_days)


def calendar_table(dates):
    """Ramadan / Hajj days for each month start in `dates`."""
    dates = pd.DatetimeIndex(sorted(set(dates)))
    return pd.DataFrame({"date": dates,
                         "ramadan_days": [days_in_month(RAMADAN, d) for d in dates],
                         "hajj_days": [days_in_month(HAJJ, d) for d in dates],
                         "prehajj_days": [days_in_month(PRE_HAJJ, d) for d in dates]})


# ---------------------------------------------------------------------------
# 4. Forecasting features (one-month-ahead)
# ---------------------------------------------------------------------------

def eligible_series(dests):
    """Series (tourist_type, destination) with 60 observed, positive counts.

    We only forecast series whose target is genuinely observed in every month:
    we never invent target values, and a reported 0 may mean "not reported".
    """
    ok = dests.groupby(["tourist_type", "destination"])["tourists"].apply(
        lambda s: s.notna().all() and (s > 0).all())
    return ok[ok].index


def build_features(dests):
    """Create the modelling table for one-month-ahead forecasts.

    Everything is on the log scale, log(1 + tourists), because counts range
    from hundreds to millions and seasonal changes are multiplicative.
    For a forecast of month t, only months t-1, t-2, ... are used, so every
    feature would really be known at the end of month t-1 (no future data).
    """
    keep = eligible_series(dests)
    df = dests.set_index(["tourist_type", "destination"]).loc[keep].reset_index()
    df = df.sort_values(["tourist_type", "destination", "date"]).reset_index(drop=True)
    df["log_y"] = np.log1p(df["tourists"].astype(float))
    g = df.groupby(["tourist_type", "destination"], observed=True)["log_y"]

    # Lags: value k months before t (shift moves values down within each series).
    for k in [1, 2, 3, 12, 13, 14, 15, 24]:
        df[f"lag{k}"] = g.shift(k)
    # Year-over-year growth seen in the most recent months (log ratios).
    df["yoy1"] = df["lag1"] - df["lag13"]
    df["yoy2"] = df["lag2"] - df["lag14"]
    df["yoy3"] = df["lag3"] - df["lag15"]
    df["yoy_mean3"] = df[["yoy1", "yoy2", "yoy3"]].mean(axis=1)
    # Month-to-month change one year earlier: how month t compared to t-1 last year.
    df["seasonal_step"] = df["lag12"] - df["lag13"]
    # Rolling mean of the last 3 observed months (uses t-1..t-3 only).
    df["roll3"] = df[["lag1", "lag2", "lag3"]].mean(axis=1)

    # Calendar features: month as a circle (Dec is next to Jan) + religious seasons.
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    cal = calendar_table(pd.date_range("2020-01-01", "2025-12-01", freq="MS"))
    cal_last_year = cal.assign(date=cal["date"] + pd.DateOffset(years=1))
    df = df.merge(cal, on="date", how="left").merge(
        cal_last_year.rename(columns={"ramadan_days": "ramadan_ly", "hajj_days": "hajj_ly",
                                      "prehajj_days": "prehajj_ly"}),
        on="date", how="left")
    # Change compared with the same month last year (copying last year would miss this).
    df["ramadan_change"] = df["ramadan_days"] - df["ramadan_ly"]
    df["hajj_change"] = df["hajj_days"] - df["hajj_ly"]
    df["prehajj_change"] = df["prehajj_days"] - df["prehajj_ly"]
    # Religious seasons matter mostly for the holy cities (Makkah & Madinah
    # provinces), so we multiply the changes by a holy-province indicator.
    holy = df["province"].isin(["Makkah", "Madinah"]).astype(int)
    for c in ["ramadan_change", "hajj_change", "prehajj_change"]:
        df[f"holy_{c}"] = holy * df[c]
    df["is_inbound"] = (df["tourist_type"] == "Inbound").astype(int)
    df["is_other"] = df["is_other"].astype(int)
    df["series"] = df["tourist_type"].astype(str) + " | " + df["destination"].astype(str)
    return df


# Feature groups used in the experiments of notebook 02.
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
    """Metrics on the ORIGINAL scale (number of tourists).

    MAE   = mean |error|                         (tourists)
    RMSE  = sqrt(mean error^2)                   (tourists, punishes big misses)
    WAPE  = sum |error| / sum actual * 100       (% of all tourists mis-forecast)
    Accuracy = 100 - WAPE                        (share of tourists forecast correctly;
                                                  NOT classification accuracy)
    R2    = 1 - SSE / SST                        (explained variation)
    MAPE  = mean(|error| / actual) * 100         (every series weighs the same)
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
