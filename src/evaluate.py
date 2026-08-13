"""
Step 3 of the pipeline: evaluate the fitted MMM the way you have to when
there is no ground truth to check against -- out-of-sample predictive
validity, posterior predictive checks, residual diagnostics, and ROAS
sanity-checking, instead of parameter-recovery MAPE against a known answer.

Loads the trace fit_model.py saved (trained on weeks 1-100 only) and the
held-out test weeks (101-117, ~14.5% of history) that were never shown to
the model, then scores it against them.
"""

import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymc_marketing.mmm import MMM

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mmm_weekly_data.csv"
TRACE_PATH = Path(__file__).resolve().parents[1] / "data" / "fitted_mmm.nc"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
METRICS_PATH = Path(__file__).resolve().parents[1] / "data" / "metrics.json"
ROAS_PATH = Path(__file__).resolve().parents[1] / "data" / "roas_estimates.csv"
RESIDUALS_PATH = Path(__file__).resolve().parents[1] / "data" / "residuals_weekly.csv"

TEST_WEEKS = 17  # must match fit_model.py's TEST_WEEKS

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

PLAUSIBLE_ROAS_RANGE = (0.3, 10.0)

CHANNEL_COLORS = {
    "spend_google_search": "#2a78d6",
    "spend_google_shopping_pmax": "#eb6834",
    "spend_meta": "#1baf7a",
    "spend_tiktok": "#eda100",
}
CHANNEL_LABELS = {
    "spend_google_search": "Google Search",
    "spend_google_shopping_pmax": "Google Shopping/PMax",
    "spend_meta": "Meta",
    "spend_tiktok": "TikTok",
}
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_COLOR = "#c3c2b7"


def rescale(posterior_var: str, idata: az.InferenceData) -> "xr.DataArray":
    """$-scale a scaled posterior variable using constant_data.target_scale (identity link)."""
    return idata.posterior[posterior_var] * idata.constant_data["target_scale"]


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - ss_res / ss_tot)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    train_df = df.iloc[:-TEST_WEEKS].reset_index(drop=True)
    test_df = df.iloc[-TEST_WEEKS:].reset_index(drop=True)

    mmm = MMM.load(str(TRACE_PATH))
    idata = mmm.idata
    target_scale = float(idata.constant_data["target_scale"].values)

    # ---- In-sample (train) posterior predictive ----
    X_train = train_df[[DATE_COLUMN] + CHANNEL_COLUMNS + CONTROL_COLUMNS]
    pp_train = mmm.sample_posterior_predictive(
        X_train, extend_idata=False, combined=False, include_last_observations=False
    )
    y_train_pred_scaled = pp_train["y"]  # dims: chain, draw, date
    y_train_pred_mean = (y_train_pred_scaled.mean(dim=["chain", "draw"]) * target_scale).values
    y_train_actual = train_df[TARGET_COLUMN].values

    train_mape = mape(y_train_actual, y_train_pred_mean)
    train_r2 = r_squared(y_train_actual, y_train_pred_mean)
    print(f"In-sample (train, n={len(train_df)}): MAPE={train_mape:.2f}%, R2={train_r2:.3f}")

    # ---- Out-of-sample (test) posterior predictive ----
    X_test = test_df[[DATE_COLUMN] + CHANNEL_COLUMNS + CONTROL_COLUMNS]
    pp_test = mmm.sample_posterior_predictive(
        X_test, extend_idata=False, combined=False, include_last_observations=True
    )
    y_test_pred_scaled = pp_test["y"]
    y_test_pred_mean = (y_test_pred_scaled.mean(dim=["chain", "draw"]) * target_scale).values
    y_test_actual = test_df[TARGET_COLUMN].values

    test_mape = mape(y_test_actual, y_test_pred_mean)
    test_r2 = r_squared(y_test_actual, y_test_pred_mean)
    print(f"Out-of-sample (test, n={len(test_df)}): MAPE={test_mape:.2f}%, R2={test_r2:.3f}")

    # ---- Convergence diagnostics ----
    conv_summary = az.summary(
        idata,
        var_names=["intercept_contribution", "adstock_alpha", "saturation_lam",
                   "saturation_beta", "gamma_control", "y_sigma"],
    )
    max_rhat = conv_summary["r_hat"].max()
    min_ess = conv_summary["ess_bulk"].min()
    n_divergences = int(idata.sample_stats["diverging"].sum())
    n_draws = idata.sample_stats.sizes["chain"] * idata.sample_stats.sizes["draw"]
    print(f"\nConvergence: max r_hat={max_rhat:.4f} (<1.01 target), "
          f"min ESS bulk={min_ess:.0f} (>400 target), "
          f"divergences={n_divergences}/{n_draws} post-tuning draws")
    failing_params = conv_summary[(conv_summary["r_hat"] >= 1.01) | (conv_summary["ess_bulk"] <= 400)]
    if len(failing_params):
        print("Parameters failing convergence targets:")
        print(failing_params.to_string())
    else:
        print("All reported parameters meet r_hat < 1.01 and ESS bulk > 400.")

    # ---- ROAS per channel, train-period spend vs train-period contribution ----
    channel_contribution_dollars = rescale("channel_contribution", idata)  # (chain, draw, date, channel)
    total_spend_train = train_df[CHANNEL_COLUMNS].sum()

    roas_rows = []
    for ch in CHANNEL_COLUMNS:
        contrib = channel_contribution_dollars.sel(channel=ch).sum(dim="date")  # (chain, draw)
        spend = total_spend_train[ch]
        roas = contrib / spend
        roas_mean = float(roas.mean())
        hdi = az.hdi(roas.values.flatten(), prob=0.89)
        plausible = PLAUSIBLE_ROAS_RANGE[0] <= roas_mean <= PLAUSIBLE_ROAS_RANGE[1]
        roas_rows.append({
            "channel": ch,
            "channel_label": CHANNEL_LABELS[ch],
            "train_spend": float(spend),
            "train_contribution_mean": float(contrib.mean()),
            "roas_mean": roas_mean,
            "roas_hdi89_lower": float(hdi[0]),
            "roas_hdi89_upper": float(hdi[1]),
            "prior_roas_benchmark": CHANNEL_ROAS_PRIOR_MEAN[ch],
            "plausible_range_flag": "OK" if plausible else "FLAG: outside plausible ROAS range",
        })
        flag = "" if plausible else "  <-- FLAGGED: outside plausible digital ROAS range"
        print(f"  {CHANNEL_LABELS[ch]}: ROAS mean={roas_mean:.2f}x, "
              f"89% HDI=[{hdi[0]:.2f}, {hdi[1]:.2f}]x{flag}")

    roas_df = pd.DataFrame(roas_rows)
    roas_df.to_csv(ROAS_PATH, index=False)

    # ---- Residual diagnostics over time ----
    residuals_df = pd.DataFrame({
        "week_start": pd.concat([train_df[DATE_COLUMN], test_df[DATE_COLUMN]]).values,
        "split": ["train"] * len(train_df) + ["test"] * len(test_df),
        "actual": np.concatenate([y_train_actual, y_test_actual]),
        "predicted": np.concatenate([y_train_pred_mean, y_test_pred_mean]),
        "us_federal_holiday_week": pd.concat(
            [train_df["us_federal_holiday_week"], test_df["us_federal_holiday_week"]]
        ).values,
        "bfcm_week": pd.concat([train_df["bfcm_week"], test_df["bfcm_week"]]).values,
    })
    residuals_df["residual"] = residuals_df["actual"] - residuals_df["predicted"]
    residuals_df["residual_pct"] = residuals_df["residual"] / residuals_df["actual"] * 100
    residuals_df.to_csv(RESIDUALS_PATH, index=False)

    holiday_mape = mape(
        residuals_df.loc[residuals_df["us_federal_holiday_week"] == 1, "actual"].values,
        residuals_df.loc[residuals_df["us_federal_holiday_week"] == 1, "predicted"].values,
    )
    non_holiday_mape = mape(
        residuals_df.loc[residuals_df["us_federal_holiday_week"] == 0, "actual"].values,
        residuals_df.loc[residuals_df["us_federal_holiday_week"] == 0, "predicted"].values,
    )
    print(f"\nMAPE on US-federal-holiday weeks: {holiday_mape:.2f}% "
          f"vs non-holiday weeks: {non_holiday_mape:.2f}%")

    # ---- metrics.json ----
    metrics = {
        "in_sample_train": {"n_weeks": len(train_df), "mape_pct": train_mape, "r2": train_r2},
        "out_of_sample_test": {"n_weeks": len(test_df), "mape_pct": test_mape, "r2": test_r2},
        "convergence": {
            "max_r_hat": float(max_rhat), "min_ess_bulk": float(min_ess),
            "n_divergences": n_divergences, "n_post_tuning_draws": int(n_draws),
            "all_targets_met": bool(len(failing_params) == 0),
        },
        "holiday_week_mape_pct": holiday_mape,
        "non_holiday_week_mape_pct": non_holiday_mape,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {METRICS_PATH} and {ROAS_PATH} and {RESIDUALS_PATH}")

    # ================= Figures =================
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": GRID,
        "axes.labelcolor": SECONDARY_INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
    })

    # ---- 01_model_fit.png: actual vs fitted + residuals ----
    hdi_train = az.hdi(y_train_pred_scaled, prob=0.89) * target_scale
    hdi_test = az.hdi(y_test_pred_scaled, prob=0.89) * target_scale

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2.2, 1]})
    all_dates = pd.concat([train_df[DATE_COLUMN], test_df[DATE_COLUMN]])
    all_actual = np.concatenate([y_train_actual, y_test_actual])
    all_pred = np.concatenate([y_train_pred_mean, y_test_pred_mean])
    lower = np.concatenate([hdi_train.sel(ci_bound="lower").values, hdi_test.sel(ci_bound="lower").values])
    upper = np.concatenate([hdi_train.sel(ci_bound="upper").values, hdi_test.sel(ci_bound="upper").values])

    ax1.fill_between(all_dates, lower, upper, color=CHANNEL_COLORS["spend_google_search"], alpha=0.15,
                      label="89% credible interval")
    ax1.plot(all_dates, all_actual, color=INK, linewidth=1.6, label="Actual revenue")
    ax1.plot(all_dates, all_pred, color=CHANNEL_COLORS["spend_google_search"], linewidth=1.6,
              linestyle="--", label="Model prediction (mean)")
    split_date = test_df[DATE_COLUMN].iloc[0]
    ax1.axvline(split_date, color=SECONDARY_INK, linewidth=1, linestyle=":")
    ax1.text(split_date, ax1.get_ylim()[1], "  train | test", va="top", fontsize=9, color=SECONDARY_INK)
    ax1.set_ylabel("Weekly revenue ($)")
    ax1.set_title("Actual vs. model-predicted weekly revenue (train + held-out test)")
    ax1.legend(loc="upper left", frameon=False, fontsize=9)

    colors = np.where(residuals_df["split"] == "test", "#e34948", MUTED)
    ax2.bar(residuals_df["week_start"], residuals_df["residual"], color=colors, width=5)
    ax2.axhline(0, color=SECONDARY_INK, linewidth=1)
    ax2.axvline(split_date, color=SECONDARY_INK, linewidth=1, linestyle=":")
    holiday_dates = residuals_df.loc[residuals_df["us_federal_holiday_week"] == 1, "week_start"]
    for d in holiday_dates:
        ax2.axvline(d, color=CHANNEL_COLORS["spend_meta"], linewidth=0.6, alpha=0.4)
    ax2.set_ylabel("Residual ($)\nactual - predicted")
    ax2.set_xlabel("Week")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_model_fit.png", dpi=150)
    plt.close(fig)

    # ---- 02_decomposition.png: stacked contributions over time (train only, the fitted period) ----
    intercept_dollars = rescale("intercept_contribution", idata).mean(dim=["chain", "draw"])
    control_dollars = rescale("control_contribution", idata).mean(dim=["chain", "draw"])  # (date, control)
    channel_dollars_mean = channel_contribution_dollars.mean(dim=["chain", "draw"])  # (date, channel)

    baseline = intercept_dollars.values + control_dollars.sum(dim="control").values
    baseline = np.clip(baseline, 0, None)  # baseline can dip slightly negative from control gammas; floor for a readable stack

    fig, ax = plt.subplots(figsize=(11, 5.5))
    stack = [baseline] + [channel_dollars_mean.sel(channel=ch).values for ch in CHANNEL_COLUMNS]
    labels = ["Baseline (intercept + trend + seasonality + holidays)"] + [CHANNEL_LABELS[c] for c in CHANNEL_COLUMNS]
    colors_stack = [BASELINE_COLOR] + [CHANNEL_COLORS[c] for c in CHANNEL_COLUMNS]
    ax.stackplot(train_df[DATE_COLUMN], *stack, labels=labels, colors=colors_stack, alpha=0.9)
    ax.set_ylabel("Weekly revenue contribution ($)")
    ax.set_title("Modeled revenue decomposition, training period (posterior mean)")
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_decomposition.png", dpi=150)
    plt.close(fig)

    # ---- 03_roas_comparison.png ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(roas_df))
    bar_colors = [CHANNEL_COLORS[c] for c in roas_df["channel"]]
    err_lower = roas_df["roas_mean"] - roas_df["roas_hdi89_lower"]
    err_upper = roas_df["roas_hdi89_upper"] - roas_df["roas_mean"]
    ax.barh(y_pos, roas_df["roas_mean"], xerr=[err_lower, err_upper], color=bar_colors,
            capsize=4, height=0.55, error_kw={"ecolor": SECONDARY_INK, "linewidth": 1.2})
    ax.scatter(roas_df["prior_roas_benchmark"], y_pos, marker="|", s=400, color=INK, zorder=5,
               label="Prior benchmark (pre-data)")
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle=":")
    ax.text(1.0, -0.65, " break-even (1x)", fontsize=8, color=MUTED, va="top")
    ax.set_ylim(-0.65, len(roas_df) - 0.35)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(roas_df["channel_label"])
    ax.set_xlabel("ROAS (posterior mean, 89% credible interval)")
    ax.set_title("Estimated ROAS by channel -- training period")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_roas_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote 3 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
