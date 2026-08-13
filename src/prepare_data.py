"""
Step 1 of the pipeline: filter, clean, and assemble the modeling dataset for
a single brand/region from the real Conjura multi-brand MMM export.

Source: Anderson (2024), "Multi-Region Marketing Mix Modeling (MMM) Dataset
for Several eCommerce Brands", figshare (Conjura). The raw file
(../data/conjura_mmm_data.csv) contains 143 daily brand-region timeseries
across 93 anonymized eCommerce brands; this script narrows that down to one
timeseries and builds a clean weekly modeling table. The same data/
directory holds both this raw input and every output this pipeline writes.

Chosen timeseries: MMM_TIMESERIES_ID 513211a5ba7d7c20145586b16abfda54
  -> a Food & Drink brand, US territory, USD. Picked (see CLAUDE.md for full
  rationale) for: >100 weeks of gap-free daily history, 4-5 well-populated
  paid channels (Google Search/Shopping/PMax, Meta, TikTok), no negative
  revenue or purchase-count anomalies, and a plausible blended ROAS (~2.7x).

Every transformation below is commented at the point it happens -- there is
no separate design doc; this file and CLAUDE.md are the documentation.
"""

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DATA_PATH = DATA_DIR / "conjura_mmm_data.csv"
OUT_DIR = DATA_DIR
OUT_PATH = OUT_DIR / "mmm_weekly_data.csv"

TARGET_TIMESERIES_ID = "513211a5ba7d7c20145586b16abfda54"

FOURIER_ORDER = 2
WEEK_LENGTH_DAYS = 7


def load_and_filter_brand(tsid: str) -> pd.DataFrame:
    """Filter the full 143-timeseries Conjura export down to one brand/region."""
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["DATE_DAY"])
    brand = df[df["MMM_TIMESERIES_ID"] == tsid].sort_values("DATE_DAY").reset_index(drop=True)
    if brand.empty:
        raise ValueError(f"No rows found for MMM_TIMESERIES_ID={tsid}")


    day_diffs = brand["DATE_DAY"].diff().dropna().unique()
    if not (len(day_diffs) == 1 and day_diffs[0] == pd.Timedelta(days=1)):
        raise AssertionError(
            "Expected a fully sequential daily series with no gaps; found "
            f"diffs={day_diffs}. Missing-day handling was not implemented "
            "because it wasn't needed for this brand -- do not silently "
            "proceed if that has changed."
        )
    return brand


def add_channel_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll the 9 raw spend columns up into 4 channels for modeling.

    NaN in a raw spend column means "no campaign spend recorded that day" for
    that specific sub-channel, not "day is missing" (every day is present --
    see load_and_filter_brand). Day-level coverage per sub-channel ranges
    continuously from 0-100% across brands, consistent with campaigns that
    simply didn't run on some days, not a tracking outage. So NaN -> $0 here,
    an explicit zero-fill, never an interpolation to a nonzero value.

    Grouping rationale (see CLAUDE.md for spend-share detail):
      - google_search: paid search alone, 100% day-coverage, cleanest channel.
      - google_shopping_pmax: Shopping + PMax + Display + Video. Shopping and
        PMax are both bottom-funnel product-intent Google inventory (PMax is
        largely Google's successor product to standalone Shopping campaigns);
        Display/Video are folded in too because their own day-coverage is too
        sparse (2%/47%) and their spend share is too small (<4% combined) to
        identify as standalone channels.
      - meta: Facebook + Instagram + Other Meta placements, combined into one
        platform-level channel.
      - tiktok: as-is, 99.6% day-coverage.
    """
    def z(col: str) -> pd.Series:
        return df[col].fillna(0.0)

    df["spend_google_search"] = z("GOOGLE_PAID_SEARCH_SPEND")
    df["spend_google_shopping_pmax"] = (
        z("GOOGLE_SHOPPING_SPEND") + z("GOOGLE_PMAX_SPEND")
        + z("GOOGLE_DISPLAY_SPEND") + z("GOOGLE_VIDEO_SPEND")
    )
    df["spend_meta"] = z("META_FACEBOOK_SPEND") + z("META_INSTAGRAM_SPEND") + z("META_OTHER_SPEND")
    df["spend_tiktok"] = z("TIKTOK_SPEND")
    return df


def add_revenue_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Revenue target = ALL_PURCHASES_ORIGINAL_PRICE - ALL_PURCHASES_GROSS_DISCOUNT,
    across all customers (not just first-time/acquisition purchases).

    This is Conjura's own "Gross Revenue" metric by construction: it nets out
    discounts the customer received, but -- per Conjura's public metric
    documentation -- it does NOT net out returns/refunds. Refund data exists
    in Conjura's platform but was not included in this anonymized export.
    So every $ revenue and ROAS figure downstream of this script is gross-of-
    returns; this is a confirmed, documented property of the data, not an
    unresolved unknown (see CLAUDE.md Limitations).

    All-customer (not first-purchase-only) revenue is used because paid media
    in this dataset -- especially Meta and Google Shopping/PMax retargeting --
    plausibly drives repeat purchases as well as acquisition, so restricting
    to first_purchases would understate the channels' true demand effect.
    """
    df["revenue"] = df["ALL_PURCHASES_ORIGINAL_PRICE"] - df["ALL_PURCHASES_GROSS_DISCOUNT"]
    return df


def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily rows to non-overlapping 7-day weeks, anchored on this
    brand's own first observed date (2021-10-15, a Friday) rather than a
    calendar week boundary, so every week is a complete 7-day sum with no
    partial-week bias. Since the daily series has zero gaps (asserted in
    load_and_filter_brand), "missing weeks" reduces to a non-issue here --
    the only real edge case is a partial trailing week, which is dropped.
    """
    first_date = df["DATE_DAY"].min()
    df = df.copy()
    df["week_index"] = (df["DATE_DAY"] - first_date).dt.days // WEEK_LENGTH_DAYS

    counts = df.groupby("week_index")["DATE_DAY"].count()
    complete_weeks = counts[counts == WEEK_LENGTH_DAYS].index
    dropped = counts[counts != WEEK_LENGTH_DAYS]
    if len(dropped):
        print(f"Dropping {len(dropped)} incomplete trailing/leading week(s): "
              f"{dropped.to_dict()} days")
    df = df[df["week_index"].isin(complete_weeks)]

    spend_cols = ["spend_google_search", "spend_google_shopping_pmax", "spend_meta", "spend_tiktok"]
    weekly = df.groupby("week_index").agg(
        week_start=("DATE_DAY", "min"),
        week_end=("DATE_DAY", "max"),
        revenue=("revenue", "sum"),
        all_purchases=("ALL_PURCHASES", "sum"),
        **{c: (c, "sum") for c in spend_cols},
    ).reset_index(drop=True)
    return weekly


def add_controls(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Build control variables from scratch -- the raw data has none.

    - trend: integer week index, 0..N-1, for a linear/flexible baseline trend.
    - yearly Fourier terms: FOURIER_ORDER sin/cos pairs at period = 52.18
      weeks (365.25/7 days), the same convention used in the prior synthetic
      project. No day-of-week / "weekly seasonality" term is included because
      day-of-week information is destroyed by aggregating to weekly grain.
    - us_federal_holiday_week: 1 if the 7-day window contains a US federal
      holiday (via the `holidays` package, US calendar, this brand's
      territory). US was chosen as the calendar because this timeseries'
      territory_name and organisation_primary_territory_name are both "US".
    - bfcm_week: 1 if the window contains Black Friday or Cyber Monday.
      Neither is a US federal holiday, but both are first-order demand
      shocks for eCommerce, so they're flagged separately rather than folded
      into the federal-holiday indicator.

    No price control is built: the only price-like fields in this dataset
    (original_price, discounts) are constructed from the same purchase rows
    as the revenue target itself, so using them as a predictor would leak
    the outcome into the regressors -- same reasoning the data dictionary
    itself gives for excluding discount rate as a control.
    """
    weekly = weekly.reset_index(drop=True).copy()
    n = len(weekly)
    weekly["trend"] = np.arange(n)

    period_weeks = 365.25 / 7
    t = np.arange(n)
    for k in range(1, FOURIER_ORDER + 1):
        weekly[f"yearly_sin_{k}"] = np.sin(2 * np.pi * k * t / period_weeks)
        weekly[f"yearly_cos_{k}"] = np.cos(2 * np.pi * k * t / period_weeks)

    us_holidays = holidays.US(years=range(weekly["week_start"].dt.year.min(),
                                           weekly["week_end"].dt.year.max() + 1))

    bfcm_dates = set()
    for yr in range(weekly["week_start"].dt.year.min(), weekly["week_end"].dt.year.max() + 1):
        thanksgiving = [d for d in us_holidays if d.year == yr and "Thanksgiving" in us_holidays[d]]
        if thanksgiving:
            black_friday = thanksgiving[0] + pd.Timedelta(days=1)
            cyber_monday = thanksgiving[0] + pd.Timedelta(days=4)
            bfcm_dates.update({black_friday, cyber_monday})

    def week_contains_any(row, date_set):
        days = pd.date_range(row["week_start"], row["week_end"], freq="D")
        return int(any(d.date() in date_set for d in days))

    holiday_dates = set(us_holidays.keys())
    weekly["us_federal_holiday_week"] = weekly.apply(
        lambda r: week_contains_any(r, holiday_dates), axis=1
    )
    weekly["bfcm_week"] = weekly.apply(
        lambda r: week_contains_any(r, bfcm_dates), axis=1
    )
    return weekly


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    brand = load_and_filter_brand(TARGET_TIMESERIES_ID)
    brand = add_channel_spend(brand)
    brand = add_revenue_target(brand)
    weekly = aggregate_to_weekly(brand)
    weekly = add_controls(weekly)

    weekly.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(weekly)} weekly rows to {OUT_PATH}")
    print(f"Date range: {weekly['week_start'].min().date()} to {weekly['week_end'].max().date()}")
    print(weekly[["week_start", "revenue", "spend_google_search",
                   "spend_google_shopping_pmax", "spend_meta", "spend_tiktok"]].describe())


if __name__ == "__main__":
    main()
