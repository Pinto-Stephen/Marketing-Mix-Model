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

# Held-out test size for the out-of-sample check performed in evaluate.py.
# This constant is duplicated (not imported) in evaluate.py so that script
# has no import-time dependency on this one; keep the two in sync.
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

# Weakly-informative, business-plausible ROAS benchmark priors.
#
# Source: these are *illustrative, generic* paid eCommerce digital-marketing
# benchmark ranges commonly cited by DTC eCommerce benchmarking aggregators
# and agencies (e.g. Triple Whale, Varos-style blended-ROAS reporting), not
# a number specific to this brand or vertical -- there is no channel-level
# ROAS ground truth available for this dataset. The relative ordering
# reflects a standard, widely-repeated pattern in that benchmark literature:
# paid search captures the highest-intent demand (highest ROAS), Shopping/
# PMax is similarly intent-driven but slightly lower, Meta blends prospecting
# and retargeting (mid ROAS), and TikTok -- generally the most top-of-funnel/
# awareness-oriented channel of the four here -- tends to run lowest.
# These are PRIOR MEANS only; the HalfNormal priors built from them are wide
# (sigma set so the prior standard deviation equals the prior mean), so the
# posterior is free to move substantially if the spend data disagrees.
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

    # HalfNormal(sigma).mean() = sigma * sqrt(2/pi) ~= 0.7979 * sigma.
    # Solve sigma so the HalfNormal's mean matches beta_mean_scaled exactly,
    # which also makes its prior std ~= mean (deliberately wide/weak).
    sigmas = [m / 0.7978845608 for m in means]
    return Prior("HalfNormal", sigma=sigmas, dims="channel")


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])

    # Time-based train/test split: fit only on the training weeks, so
    # evaluate.py can score genuine out-of-sample predictions on the held-out
    # tail rather than in-sample fit alone. The model saved by this script
    # (and therefore every downstream attribution/ROAS/budget-optimizer
    # number) is fit on the training period only, 2021-10 through the split
    # date below -- see CLAUDE.md for why this is the honest choice given we
    # have no ground truth to validate against otherwise.
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
    # Priors are derived from the training data's own max spend/revenue, not
    # the full dataset, so no information from the held-out test weeks leaks
    # into the model via the priors either.
    saturation_beta_prior = build_saturation_beta_prior(train_df)

    model_config = {
        "saturation_beta": saturation_beta_prior,
        # adstock_alpha (decay) and saturation_lam (steepness) are left at
        # pymc-marketing's weakly-informative defaults -- Beta(1,3) and
        # Gamma(3,1) respectively -- because we have no lift-test or other
        # exogenous signal to move them away from generic weak priors.
    }

    mmm = MMM(
        date_column=DATE_COLUMN,
        channel_columns=CHANNEL_COLUMNS,
        control_columns=CONTROL_COLUMNS,
        target_column=TARGET_COLUMN,
        adstock=GeometricAdstock(l_max=8),  # up to 8 weeks (~2 months) of carryover
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

    # Original-$-scale contributions (for ROAS etc.) are NOT computed here via
    # add_original_scale_contribution_variable: that method inserts new
    # pm.Deterministic nodes into the PyMC model, but since it's called after
    # mmm.fit() has already sampled, those nodes are never backfilled into the
    # saved trace -- they'd be silently absent from fitted_mmm.nc. Instead,
    # evaluate.py rescales channel_contribution/intercept_contribution/
    # control_contribution to $ terms itself, by multiplying the (identity-
    # link) scaled posterior by idata.constant_data["target_scale"], which is
    # exactly what that method would have done had it taken effect.
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
