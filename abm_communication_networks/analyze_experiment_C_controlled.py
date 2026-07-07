"""
analyze_experiment_C_controlled.py
------------------------------------
Experiment C addendum — mobility-controlled follow-up.

The original Experiment C (urban_light vs urban_medium) confounds mobility
with load (20 vs 30 users, 3 vs 4 base stations). This script isolates
mobility alone: urban_medium (30 users, low mobility) vs urban_controlled
(30 users, high mobility) — identical topology, app mix, and CR budget,
only user_mobility differs.

Reuses the eval/trained/{scenario}/s{seed}/ logs already produced by
generate_results.py — no new simulation runs beyond the urban_controlled
scenario are needed.

Usage
-----
    python analyze_experiment_C_controlled.py

Outputs (in resultats/)
------------------------
    expC_controlled_satisfaction_comparison.png
    expC_controlled_placement_volatility.png
    expC_controlled_summary.csv
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_ROOT = "resultats"
SCENARIOS    = ["urban_medium", "urban_controlled"]
SEEDS        = list(range(20))
WARMUP       = 50
DPI          = 300

SCENARIO_LABELS = {
    "urban_medium":     "Urban-Medium\n(low mobility, 30 users)",
    "urban_controlled": "Urban-Controlled\n(high mobility, 30 users)",
}
SCENARIO_COLORS = {
    "urban_medium":     "#3498db",
    "urban_controlled": "#9b59b6",
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 100})


def _ci95(values):
    n = len(values)
    return 0.0 if n < 2 else 1.96 * float(np.std(values, ddof=1)) / np.sqrt(n)


def _eval_dir(scenario, seed):
    return os.path.join(RESULTS_ROOT, f"resultats-{scenario}", "logs", "eval", "trained", f"s{seed}")


def _collect_satisfaction(scenario):
    means = []
    for seed in SEEDS:
        path = os.path.join(_eval_dir(scenario, seed), "satisfaction_summary.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df["Step"] > WARMUP]
        if not df.empty:
            means.append(float(df["Rate_Global"].mean()))
    return means


def _placement_changes(scenario):
    mean_changes, volatile_frac = [], []
    for seed in SEEDS:
        path = os.path.join(_eval_dir(scenario, seed), "cr_utilization.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df["Step"] > WARMUP]
        if df.empty:
            continue
        hosts_per_step = (
            df.groupby("Step")["BS_ID"]
            .apply(lambda s: frozenset(s.tolist()))
            .sort_index()
        )
        sets = hosts_per_step.to_numpy()
        if len(sets) < 2:
            continue
        changes = [len(sets[i].symmetric_difference(sets[i - 1])) for i in range(1, len(sets))]
        mean_changes.append(float(np.mean(changes)))
        volatile_frac.append(float(np.mean([c > 0 for c in changes])))
    return mean_changes, volatile_frac


def fig_satisfaction_comparison(all_sat, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = np.arange(len(SCENARIOS))
    means = [np.mean(all_sat[sc]) if all_sat[sc] else 0.0 for sc in SCENARIOS]
    cis = [_ci95(all_sat[sc]) for sc in SCENARIOS]
    colors = [SCENARIO_COLORS[sc] for sc in SCENARIOS]

    bars = ax.bar(x, means, yerr=cis, capsize=7, color=colors, alpha=0.87,
                  edgecolor="white", linewidth=1.2,
                  error_kw={"elinewidth": 2, "ecolor": "#333333"})
    for bar, m, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, m + ci + 0.015,
                f"{m:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(1.0, color="grey", linestyle="--", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[sc] for sc in SCENARIOS], fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Mean global satisfaction rate (post-warmup)")
    ax.set_title("Experiment C (controlled) — Mobility effect at fixed load (30 users)",
                 fontweight="bold")

    fig.tight_layout()
    path = os.path.join(out_dir, "expC_controlled_satisfaction_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def fig_placement_volatility(all_changes, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(SCENARIOS))
    colors = [SCENARIO_COLORS[sc] for sc in SCENARIOS]

    ax = axes[0]
    means = [np.mean(all_changes[sc][0]) if all_changes[sc][0] else 0.0 for sc in SCENARIOS]
    cis   = [_ci95(all_changes[sc][0]) for sc in SCENARIOS]
    ax.bar(x, means, yerr=cis, capsize=6, color=colors, alpha=0.87,
           edgecolor="white", linewidth=1.2,
           error_kw={"elinewidth": 2, "ecolor": "#333333"})
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[sc] for sc in SCENARIOS], fontsize=10)
    ax.set_ylabel("Mean # BS switching CR assignment / step")
    ax.set_title("Placement change magnitude", fontweight="bold")

    ax = axes[1]
    means = [np.mean(all_changes[sc][1]) if all_changes[sc][1] else 0.0 for sc in SCENARIOS]
    cis   = [_ci95(all_changes[sc][1]) for sc in SCENARIOS]
    ax.bar(x, means, yerr=cis, capsize=6, color=colors, alpha=0.87,
           edgecolor="white", linewidth=1.2,
           error_kw={"elinewidth": 2, "ecolor": "#333333"})
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[sc] for sc in SCENARIOS], fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Share of steps with >=1 placement change")
    ax.set_title("Placement change frequency", fontweight="bold")

    fig.suptitle("Experiment C (controlled) — CR placement volatility at fixed load",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "expC_controlled_placement_volatility.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def write_summary_csv(all_sat, all_changes, out_dir):
    path = os.path.join(out_dir, "expC_controlled_summary.csv")
    rows = []
    for sc in SCENARIOS:
        sat = all_sat[sc]
        changes, volatile = all_changes[sc]
        rows.append({
            "scenario": sc,
            "rate_global_mean": round(np.mean(sat), 4) if sat else None,
            "rate_global_ci95": round(_ci95(sat), 4) if sat else None,
            "mean_bs_switched_per_step": round(np.mean(changes), 4) if changes else None,
            "mean_bs_switched_ci95": round(_ci95(changes), 4) if changes else None,
            "volatile_step_frac_mean": round(np.mean(volatile), 4) if volatile else None,
            "volatile_step_frac_ci95": round(_ci95(volatile), 4) if volatile else None,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  -> {path}")


def main():
    print("Experiment C (controlled) — mobility isolated at fixed load (urban_medium vs urban_controlled)")
    all_sat = {}
    all_changes = {}
    for sc in SCENARIOS:
        all_sat[sc] = _collect_satisfaction(sc)
        all_changes[sc] = _placement_changes(sc)
        n_seeds = len(all_sat[sc])
        if n_seeds < len(SEEDS):
            print(f"  [WARNING] {sc}: only {n_seeds}/{len(SEEDS)} seeds found.")

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    fig_satisfaction_comparison(all_sat, RESULTS_ROOT)
    fig_placement_volatility(all_changes, RESULTS_ROOT)
    write_summary_csv(all_sat, all_changes, RESULTS_ROOT)
    print("Done.")


if __name__ == "__main__":
    main()
