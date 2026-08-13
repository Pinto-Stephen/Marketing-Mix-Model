"""
Step 2 of the pipeline: fit a Bayesian MMM to the real, weekly Conjura data
prepared by prepare_data.py.

Unlike the prior synthetic-recovery project, there is no lift-test data for
this brand and none is fabricated (fabricating one would defeat the entire
point of validating against real, unlabeled data). Instead the
adstock/saturation priors are set to be weakly-informative but centered on
published paid-media ROAS benchmarks -- see `channel_roas_priors` below for
the exact source and numbers. This is a real limitation, not a bug: without
incrementality tests, adstock decay (alpha) and saturation steepness (lam)
are only weakly identified from observational spend data alone, because
spend across channels here is correlated (most channels ramp/dip together
with overall marketing pressure and seasonality) and adstock/saturation
shape can trade off against each other to fit the same spend->revenue curve
in more than one way. See CLAUDE.md "Limitations" for the full discussion.
"""

from pathlib import Path

import arviz as az
import pandas as pd
from pymc_extras.prior import Prior
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mmm_weekly_data.csv"
TRACE_PATH = Path(__file__).resolve().parents[1] / "data" / "fitted_mmm.nc"
CONVERGENCE_PATH = Path(__file__).resolve().parents[1] / "data" / "convergence_summary.csv"


TEST_WEEKS = 17  # ~14.5% of 117 weeks, in the 10-15% range requested

DATE_COLUMN = "week_start"
TARGET_COLUMN = "revenue"
CHANNEL_COLUMNS = [
    "spend_google_search",
    "spend_google_shopping_pmax",
    "spend_meta",
    "spend_tiktok",
]
CONTROL_COLUMNS = [
    "trend",
    "yearly_sin_1", "yearly_cos_1",
    "yearly_sin_2", "yearly_cos_2",
    "us_federal_holiday_week",
    "bfcm_week",
]


CHANNEL_ROAS_PRIOR_MEAN = {
    "spend_google_search": 4.0,
    "spend_google_shopping_pmax": 3.0,
    "spend_meta": 2.5,
    "spend_tiktok": 2.0,
}


def build_saturation_beta_prior(df: pd.DataFrame) -> Prior:
    """
    Translate the ROAS benchmark priors above into a `saturation_beta` prior
    in the model's internal *scaled* units.

    MMM's default scaling (confirmed via `mmm.scaling`) is per-channel max-
    scaling for spend (x_scaled = x / max(x) for that channel) and a single
    global max-scaling for the target (y_scaled = revenue / max(revenue)).
    `saturation_beta` is the asymptote of `beta * (1 - exp(-lam * x_scaled))`
    as x_scaled -> infinity, i.e. the scaled contribution ceiling.

    We treat "spend at its historical weekly maximum, converting at the
    benchmark ROAS" as a reasonable proxy for that ceiling:

        beta_prior_mean_scaled = ROAS_benchmark * max(spend_channel) / max(revenue)

    This is an approximation (the true contribution at max spend is somewhat
    below the asymptote, since saturation hasn't fully saturated by
    x_scaled=1), which if anything makes this prior mean conservative
    (understates beta slightly) rather than overconfident.
    """
    max_revenue = df[TARGET_COLUMN].max()
    means = []
    for ch in CHANNEL_COLUMNS:
        max_spend = df[ch].max()
        roas = CHANNEL_ROAS_PRIOR_MEAN[ch]
        beta_mean_scaled = roas * max_spend / max_revenue
        means.append(beta_mean_scaled)
        print(f"  {ch}: ROAS prior mean={roas}x, max_spend={max_spend:,.0f}, "
              f"-> saturation_beta prior mean (scaled)={beta_mean_scaled:.3f}")

    sigmas = [m / 0.7978845608 for m in means]
    return Prior("HalfNormal", sigma=sigmas, dims="channel")


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])


    n_test = TEST_WEEKS
    train_df = df.iloc[:-n_test].reset_index(drop=True)
    test_df = df.iloc[-n_test:].reset_index(drop=True)
    print(f"Train: {len(train_df)} weeks ({train_df[DATE_COLUMN].min().date()} to "
          f"{train_df[DATE_COLUMN].max().date()})")
    print(f"Test (held out, not used to fit): {len(test_df)} weeks "
          f"({test_df[DATE_COLUMN].min().date()} to {test_df[DATE_COLUMN].max().date()})")

    X = train_df[[DATE_COLUMN] + CHANNEL_COLUMNS + CONTROL_COLUMNS]
    y = train_df[TARGET_COLUMN]

    print("\nDeriving business-plausible saturation_beta priors from ROAS benchmarks:")

    saturation_beta_prior = build_saturation_beta_prior(train_df)

    model_config = {
        "saturation_beta": saturation_beta_prior,
    }

    mmm = MMM(
        date_column=DATE_COLUMN,
        channel_columns=CHANNEL_COLUMNS,
        control_columns=CONTROL_COLUMNS,
        target_column=TARGET_COLUMN,
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        model_config=model_config,
    )

    print("\nSampling (4 chains, 1000 tune + 1000 draws, target_accept=0.9)...")
    mmm.fit(
        X, y,
        chains=4,
        draws=1000,
        tune=1000,
        target_accept=0.9,
        random_seed=42,
    )

    mmm.save(str(TRACE_PATH))
    print(f"\nSaved fitted trace to {TRACE_PATH}")

    summary = az.summary(
        mmm.idata,
        var_names=["intercept_contribution", "adstock_alpha", "saturation_lam",
                   "saturation_beta", "gamma_control", "y_sigma"],
    )
    summary.to_csv(CONVERGENCE_PATH)
    print(f"\nConvergence summary (also written to {CONVERGENCE_PATH}):")
    print(summary.to_string())

    max_rhat = summary["r_hat"].max()
    min_ess = summary["ess_bulk"].min()
    print(f"\nmax r_hat = {max_rhat:.4f} (target < 1.01), min ESS bulk = {min_ess:.0f} (target > 400)")
    if max_rhat >= 1.01 or min_ess <= 400:
        print("WARNING: convergence diagnostics are outside standard targets -- "
              "see evaluate.py output / CLAUDE.md for how this is handled.")


if __name__ == "__main__":
    main()
