"""
Dissertation-quality plots for abm_communication_networks experiments.

Usage
-----
    python plots/plot_results.py --logs-dir logs/ --output-dir plots/

Expected directory layout
-------------------------
    logs/
      train_rn/
        rn_reward.csv          (Step, TotalReward)
      train_cr/
        cr_reward.csv          (Step, SatisfactionRate)
        cr_delta_q.csv         (Step, DeltaQ)
      eval/
        {strategy}/            one of: random | static | trained | centralized
          {seed_or_scenario}/  any subdirectory structure
            satisfaction_summary.csv
        trained/
          {scenario}/          urban_light | urban_medium | urban_dense
            {seed}/
              satisfaction_summary.csv

Figures produced
----------------
    fig1_training_curves.png
    fig2_delta_q.png
    fig3_strategy_comparison.png
    fig4_per_app_satisfaction.png
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stats_utils import ci95 as _ci95

# ── Global style ─────────────────────────────────────────────────────────────

plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "figure.dpi":       100,   # screen preview; overridden by savefig dpi=300
})

DPI            = 300
WINDOW         = 50     # moving-average window (steps)
CONVERGENCE_DQ = 0.01   # horizontal threshold line for Fig. 2

STRATEGY_COLORS = {
    "random":      "#e74c3c",
    "static":      "#f39c12",
    "trained":     "#2980b9",
    "centralized": "#27ae60",
}
STRATEGY_LABELS = {
    "random":      "Random",
    "static":      "Static",
    "trained":     "Trained (RL)",
    "centralized": "Centralized",
}
SCENARIO_COLORS = {
    "urban_light":  "#2ecc71",
    "urban_medium": "#3498db",
    "urban_dense":  "#e67e22",
}
SCENARIO_LABELS = {
    "urban_light":  "Urban-Light",
    "urban_medium": "Urban-Medium",
    "urban_dense":  "Urban-Dense",
}
APP_LABELS = {
    "Rate_AR_VR":       "AR/VR",
    "Rate_Streaming":   "Streaming",
    "Rate_BestEffort":  "Best-effort",
}

# ── Utilities ─────────────────────────────────────────────────────────────────

def _warn(msg: str) -> None:
    print(f"[WARNING] {msg}", file=sys.stderr)


def _load(path: str, required_cols=None) -> pd.DataFrame | None:
    """Return DataFrame or None (with warning) if file is missing / malformed."""
    if not os.path.exists(path):
        _warn(f"File not found, skipping: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        _warn(f"Could not read {path}: {exc}")
        return None
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            _warn(f"{path}: missing columns {missing}, skipping.")
            return None
    return df


def _moving_avg(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, center=True, min_periods=1).mean()


def _collect_satisfaction(strategy_dir: str, warmup: int) -> list[float]:
    """Walk *strategy_dir*, gather per-seed mean global satisfaction (post-warmup)."""
    results = []
    for root, _, files in os.walk(strategy_dir):
        if "satisfaction_summary.csv" not in files:
            continue
        df = _load(os.path.join(root, "satisfaction_summary.csv"),
                   required_cols=["Step", "Rate_Global"])
        if df is None:
            continue
        df = df[df["Step"] > warmup]
        if not df.empty:
            results.append(float(df["Rate_Global"].mean()))
    return results


def _collect_per_app(scenario_dir: str, warmup: int) -> dict[str, list[float]]:
    """Walk *scenario_dir*, gather per-seed means for each app type."""
    cols = list(APP_LABELS.keys())
    data: dict[str, list[float]] = {c: [] for c in cols}
    for root, _, files in os.walk(scenario_dir):
        if "satisfaction_summary.csv" not in files:
            continue
        df = _load(os.path.join(root, "satisfaction_summary.csv"),
                   required_cols=["Step"] + cols)
        if df is None:
            continue
        df = df[df["Step"] > warmup]
        if df.empty:
            continue
        for col in cols:
            data[col].append(float(df[col].mean()))
    return data


# ── Figure 1: Training curves ────────────────────────────────────────────────

def fig1_training_curves(logs_dir: str, output_dir: str) -> None:
    rn_df = _load(os.path.join(logs_dir, "train_rn", "rn_reward.csv"),
                  ["Step", "TotalReward"])
    cr_df = _load(os.path.join(logs_dir, "train_cr", "cr_reward.csv"),
                  ["Step", "SatisfactionRate"])

    if rn_df is None and cr_df is None:
        _warn("No training data for Fig. 1, skipping.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle(
        "Training convergence — RN agent (top) / CR agent (bottom)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # ── Subplot 1: RN reward ──────────────────────────────────────────
    ax = axes[0]
    if rn_df is not None:
        x, y = rn_df["Step"], rn_df["TotalReward"]
        ax.plot(x, y, color="lightgrey", lw=1, label="Raw reward")
        ax.plot(x, _moving_avg(y, WINDOW), color="steelblue", lw=2,
                label=f"Moving avg  (w = {WINDOW})")
        ax.set_ylabel("Total reward  (users connected to RN)", fontsize=12)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "rn_reward.csv not found", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    ax.set_title("RN Q-learning agent", fontsize=12)
    ax.set_xlabel("Training step", fontsize=12)

    # ── Subplot 2: CR satisfaction ────────────────────────────────────
    ax = axes[1]
    if cr_df is not None:
        x, y = cr_df["Step"], cr_df["SatisfactionRate"]
        ax.plot(x, y, color="lightgrey", lw=1, label="Raw satisfaction rate")
        ax.plot(x, _moving_avg(y, WINDOW), color="darkorange", lw=2,
                label=f"Moving avg  (w = {WINDOW})")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylabel("Satisfaction rate", fontsize=12)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "cr_reward.csv not found", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    ax.set_title("CR placement Q-learning agent", fontsize=12)
    ax.set_xlabel("Training step", fontsize=12)

    fig.tight_layout()
    out = os.path.join(output_dir, "fig1_training_curves.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: Delta-Q convergence ────────────────────────────────────────────

def fig2_delta_q(logs_dir: str, output_dir: str) -> None:
    df = _load(os.path.join(logs_dir, "train_cr", "cr_delta_q.csv"),
               ["Step", "DeltaQ"])
    if df is None:
        _warn("cr_delta_q.csv not found — re-run train_cr to generate it. Skipping Fig. 2.")
        return

    x, y = df["Step"], df["DeltaQ"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color="lightgrey", lw=1, label="Raw |ΔQ|", zorder=1)
    ax.plot(x, _moving_avg(y, WINDOW), color="crimson", lw=2,
            label=f"Moving avg  (w = {WINDOW})", zorder=2)
    ax.axhline(CONVERGENCE_DQ, color="navy", linestyle="--", lw=1.5,
               label=f"Convergence threshold  (ΔQ = {CONVERGENCE_DQ})", zorder=3)

    ax.set_yscale("log")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("|ΔQ|  (log scale)", fontsize=12)
    ax.set_title("Q-value convergence — CR placement agent",
                 fontsize=14, fontweight="bold")
    ax.legend()

    fig.tight_layout()
    out = os.path.join(output_dir, "fig2_delta_q.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Strategy comparison ────────────────────────────────────────────

def fig3_strategy_comparison(
    logs_dir: str, output_dir: str, warmup: int
) -> None:
    eval_dir = os.path.join(logs_dir, "eval")
    strategies = ["random", "static", "trained", "centralized"]

    means, cis, valid = [], [], []
    for strat in strategies:
        strat_dir = os.path.join(eval_dir, strat)
        if not os.path.isdir(strat_dir):
            _warn(f"Strategy folder not found: {strat_dir}")
            continue
        seed_means = _collect_satisfaction(strat_dir, warmup)
        if not seed_means:
            _warn(f"No satisfaction data for '{strat}', skipping.")
            continue
        if len(seed_means) == 1:
            _warn(f"Only 1 seed for '{strat}' — CI = 0 (run more seeds for intervals).")
        means.append(np.mean(seed_means))
        cis.append(_ci95(seed_means))
        valid.append(strat)

    if not valid:
        _warn("No strategy data found for Fig. 3, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(valid))
    colors = [STRATEGY_COLORS[s] for s in valid]
    xlabels = [STRATEGY_LABELS[s] for s in valid]

    bars = ax.bar(
        x, means, yerr=cis, capsize=7,
        color=colors, alpha=0.85, edgecolor="white", linewidth=1.2,
        error_kw={"elinewidth": 2, "ecolor": "#333333", "capthick": 2},
    )

    for bar, m, ci in zip(bars, means, cis):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + ci + 0.015,
            f"{m:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.axhline(1.0, color="grey", linestyle="--", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Mean satisfaction rate", fontsize=12)
    ax.set_title(
        "Placement strategy comparison — mean satisfaction ± 95 % CI",
        fontsize=14, fontweight="bold",
    )

    fig.tight_layout()
    out = os.path.join(output_dir, "fig3_strategy_comparison.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: Per-app satisfaction ───────────────────────────────────────────

def fig4_per_app_satisfaction(
    logs_dir: str, output_dir: str, warmup: int
) -> None:
    trained_dir = os.path.join(logs_dir, "eval", "trained")
    scenarios = ["urban_light", "urban_medium", "urban_dense"]
    app_cols = list(APP_LABELS.keys())

    scenario_data: dict[str, dict] = {}
    for sc in scenarios:
        sc_dir = os.path.join(trained_dir, sc)
        if not os.path.isdir(sc_dir):
            _warn(f"Scenario folder not found: {sc_dir}")
            continue
        data = _collect_per_app(sc_dir, warmup)
        if all(len(v) == 0 for v in data.values()):
            _warn(f"No data for scenario '{sc}', skipping.")
            continue
        scenario_data[sc] = data

    if not scenario_data:
        _warn("No per-app data found for Fig. 4, skipping.")
        return

    n_apps = len(app_cols)
    n_sc   = len(scenario_data)
    group_w = 0.75
    bar_w   = group_w / n_sc
    x = np.arange(n_apps)

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, (sc, data) in enumerate(scenario_data.items()):
        offsets = x + (idx - n_sc / 2 + 0.5) * bar_w
        m   = [np.mean(data[c]) if data[c] else 0.0 for c in app_cols]
        ci  = [_ci95(data[c]) for c in app_cols]
        ax.bar(
            offsets, m, yerr=ci, width=bar_w * 0.88,
            color=SCENARIO_COLORS[sc], alpha=0.85, edgecolor="white",
            linewidth=1, capsize=5, label=SCENARIO_LABELS.get(sc, sc),
            error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([APP_LABELS[c] for c in app_cols], fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Mean satisfaction rate", fontsize=12)
    ax.set_title(
        "Per-application satisfaction — trained CR agent",
        fontsize=14, fontweight="bold",
    )
    ax.axhline(1.0, color="grey", linestyle="--", lw=1, alpha=0.5)
    ax.legend(title="Scenario", fontsize=11, title_fontsize=11)

    fig.tight_layout()
    out = os.path.join(output_dir, "fig4_per_app_satisfaction.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dissertation-quality plots from experiment CSV logs."
    )
    parser.add_argument(
        "--logs-dir", default="logs/",
        help="Root log directory containing train_rn/, train_cr/, eval/ "
             "(default: logs/)",
    )
    parser.add_argument(
        "--output-dir", default="plots/",
        help="Directory for PNG output (default: plots/)",
    )
    parser.add_argument(
        "--warmup", type=int, default=50,
        help="Steps excluded from eval metrics (default: 50)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Logs    : {args.logs_dir}")
    print(f"Output  : {args.output_dir}")
    print(f"Warmup  : {args.warmup} steps\n")

    print("Figure 1 — Training curves")
    fig1_training_curves(args.logs_dir, args.output_dir)

    print("\nFigure 2 — Delta-Q convergence")
    fig2_delta_q(args.logs_dir, args.output_dir)

    print("\nFigure 3 — Strategy comparison")
    fig3_strategy_comparison(args.logs_dir, args.output_dir, args.warmup)

    print("\nFigure 4 — Per-app satisfaction (trained agent)")
    fig4_per_app_satisfaction(args.logs_dir, args.output_dir, args.warmup)

    print(f"\nDone. Plots in '{args.output_dir}'.")


if __name__ == "__main__":
    main()
