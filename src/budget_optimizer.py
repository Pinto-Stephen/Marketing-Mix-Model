"""
Step 4 of the pipeline: a DIRECTIONAL budget-reallocation suggestion using
PyMC-Marketing's BudgetOptimizer.

"Directional" is the operative word -- see CLAUDE.md Limitations. The
adstock/saturation curves this optimizer walks were fit on ~2 years of
observational spend with no incrementality-test calibration, and the ROAS
estimates in evaluate.py already showed two of the four channels
(google_shopping_pmax, meta) with posterior means below break-even and wide
credible intervals. An optimizer built on those curves will happily push
budget hard toward whichever channel the model currently believes is most
efficient (google_search here) -- that conclusion is only as trustworthy as
the underlying attribution, which this project has been explicit about not
fully trusting. This script reports what the model recommends, not a
production budget decision.

There are no real future dates in this dataset, so the optimization window
is a synthetic 4-week period immediately following the last observed week
(2024-01-12 to 2024-02-02) -- a typical near-term reallocation planning
horizon. Per-channel bounds are set to [30%, 250%] of each channel's most
recent actual weekly spend, deliberately excluding the corner solution
(all-in on one channel) an unconstrained optimizer would pick given the
weakly-identified posterior -- see the printed unconstrained comparison too.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.mmm import BudgetOptimizerWrapper

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mmm_weekly_data.csv"
TRACE_PATH = Path(__file__).resolve().parents[1] / "data" / "fitted_mmm.nc"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "budget_reallocation.json"

CHANNEL_COLUMNS = [
    "spend_google_search",
    "spend_google_shopping_pmax",
    "spend_meta",
    "spend_tiktok",
]
CHANNEL_LABELS = {
    "spend_google_search": "Google Search",
    "spend_google_shopping_pmax": "Google Shopping/PMax",
    "spend_meta": "Meta",
    "spend_tiktok": "TikTok",
}
CHANNEL_COLORS = {
    "spend_google_search": "#2a78d6",
    "spend_google_shopping_pmax": "#eb6834",
    "spend_meta": "#1baf7a",
    "spend_tiktok": "#eda100",
}

RECENT_WEEKS_FOR_BASELINE = 4
OPTIMIZATION_START = "2024-01-12"
OPTIMIZATION_END = "2024-02-02"
BOUND_LOWER_MULT = 0.3
BOUND_UPPER_MULT = 2.5


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    mmm = MMM.load(str(TRACE_PATH))
    df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])

    recent = df.iloc[-RECENT_WEEKS_FOR_BASELINE:]
    current_weekly_spend = recent[CHANNEL_COLUMNS].mean()
    print(f"'Current' allocation = avg weekly spend, last {RECENT_WEEKS_FOR_BASELINE} actual weeks "
          f"({recent['week_start'].min().date()} to {recent['week_start'].max().date()}):")
    print(current_weekly_spend.to_string())

    wrapper = BudgetOptimizerWrapper(model=mmm, start_date=OPTIMIZATION_START, end_date=OPTIMIZATION_END)
    n_periods = wrapper.num_periods
    total_budget = float(current_weekly_spend.sum() * n_periods)
    print(f"\nOptimization window: {OPTIMIZATION_START} to {OPTIMIZATION_END} "
          f"({n_periods} weeks), total budget held fixed at ${total_budget:,.0f}")

    # ---- Status quo: pin every channel to its current level (bounds collapsed to a point) ----
    status_quo_bounds = {ch: (current_weekly_spend[ch] * n_periods, current_weekly_spend[ch] * n_periods)
                          for ch in CHANNEL_COLUMNS}
    sq_alloc, sq_res = wrapper.optimize_budget(budget=total_budget, budget_bounds=status_quo_bounds)
    status_quo_response = float(-sq_res.fun)

    # ---- Bounded optimization: [30%, 250%] of current weekly spend per channel ----
    bounded_bounds = {ch: (BOUND_LOWER_MULT * current_weekly_spend[ch] * n_periods,
                            BOUND_UPPER_MULT * current_weekly_spend[ch] * n_periods)
                       for ch in CHANNEL_COLUMNS}
    bounded_alloc, bounded_res = wrapper.optimize_budget(budget=total_budget, budget_bounds=bounded_bounds)
    bounded_response = float(-bounded_res.fun)

    # ---- Unconstrained (0, total_budget) optimization, for comparison only ----
    unconstrained_bounds = {ch: (0.0, total_budget) for ch in CHANNEL_COLUMNS}
    unc_alloc, unc_res = wrapper.optimize_budget(budget=total_budget, budget_bounds=unconstrained_bounds)
    unconstrained_response = float(-unc_res.fun)

    print(f"\nStatus quo expected {n_periods}-week revenue:      ${status_quo_response:,.0f}")
    print(f"Bounded-optimal expected {n_periods}-week revenue:  ${bounded_response:,.0f}  "
          f"({(bounded_response / status_quo_response - 1) * 100:+.1f}%)")
    print(f"Unconstrained-optimal expected {n_periods}-week rev: ${unconstrained_response:,.0f}  "
          f"({(unconstrained_response / status_quo_response - 1) * 100:+.1f}%)  [illustrative only, see caveats]")

    print("\nBounded-optimal weekly-equivalent allocation vs current:")
    result_rows = []
    for ch in CHANNEL_COLUMNS:
        current_wk = float(current_weekly_spend[ch])
        optimal_wk = float(bounded_alloc.sel(channel=ch).values) / n_periods
        unconstrained_wk = float(unc_alloc.sel(channel=ch).values) / n_periods
        pct_change = (optimal_wk / current_wk - 1) * 100
        print(f"  {CHANNEL_LABELS[ch]}: ${current_wk:,.0f}/wk -> ${optimal_wk:,.0f}/wk ({pct_change:+.1f}%)")
        result_rows.append({
            "channel": ch,
            "channel_label": CHANNEL_LABELS[ch],
            "current_weekly_spend": current_wk,
            "bounded_optimal_weekly_spend": optimal_wk,
            "pct_change": pct_change,
            "unconstrained_optimal_weekly_spend": unconstrained_wk,
        })

    output = {
        "optimization_window": {"start": OPTIMIZATION_START, "end": OPTIMIZATION_END, "n_periods": n_periods},
        "total_budget_held_fixed": total_budget,
        "bounds_used": f"[{BOUND_LOWER_MULT}x, {BOUND_UPPER_MULT}x] of current weekly spend per channel",
        "status_quo_expected_response": status_quo_response,
        "bounded_optimal_expected_response": bounded_response,
        "bounded_uplift_pct": (bounded_response / status_quo_response - 1) * 100,
        "unconstrained_optimal_expected_response": unconstrained_response,
        "unconstrained_uplift_pct": (unconstrained_response / status_quo_response - 1) * 100,
        "channels": result_rows,
        "caveat": (
            "This allocation is directional only. It is derived from adstock/saturation "
            "curves fit on observational spend with no incrementality-test calibration; "
            "evaluate.py's ROAS estimates already show wide credible intervals and two "
            "channels (google_shopping_pmax, meta) with posterior-mean ROAS below "
            "break-even. Treat this as a hypothesis to test (e.g. via a small geo-holdout "
            "or budget-shift experiment), not an instruction."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {OUT_PATH}")

    # ---- Figure ----
    plt.rcParams.update({
        "font.family": "sans-serif", "axes.edgecolor": "#e1e0d9", "axes.labelcolor": "#52514e",
        "text.color": "#0b0b0b", "xtick.color": "#898781", "ytick.color": "#898781",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(CHANNEL_COLUMNS))
    bar_h = 0.35
    current_vals = [current_weekly_spend[ch] for ch in CHANNEL_COLUMNS]
    optimal_vals = [float(bounded_alloc.sel(channel=ch).values) / n_periods for ch in CHANNEL_COLUMNS]

    ax.barh(y_pos + bar_h / 2, current_vals, height=bar_h, color="#c3c2b7", label="Current (avg weekly spend)")
    ax.barh(y_pos - bar_h / 2, optimal_vals, height=bar_h,
            color=[CHANNEL_COLORS[c] for c in CHANNEL_COLUMNS], label="Bounded-optimal (weekly-equivalent)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([CHANNEL_LABELS[c] for c in CHANNEL_COLUMNS])
    ax.set_xlabel("Weekly spend ($)")
    ax.set_title(f"Directional budget reallocation (bounds: {BOUND_LOWER_MULT}x-{BOUND_UPPER_MULT}x current spend)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_budget_reallocation.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIG_DIR / '04_budget_reallocation.png'}")


if __name__ == "__main__":
    main()
