"""
paired_diff_experimentA.py
----------------------------
Paired-seed difference analysis for Experiment A (Section 7.2 / tab:expA).

All five strategies in Experiment A are evaluated on the same seeds, so the
same seed produces identical user trajectories, initial positions, and
shadow-fading realisations regardless of which strategy is being scored
(dissertation Sec. 6.3.3). Comparing independent 95% CIs per strategy (as in
fig3_strategy_comparison.png) throws away this pairing and is overly
conservative. This script instead computes, seed by seed, the difference in
post-warmup mean satisfaction between the RL agent and each baseline:

    diff_i = satisfaction_RL(seed_i) - satisfaction_X(seed_i)

and reports mean_diff +/- 1.96 * std(diff) / sqrt(n), a paired CI that
cancels the seed-to-seed noise shared by both strategies.

Reuses the eval logs already produced by generate_results.py -- no new
simulation runs.

Usage
-----
    python paired_diff_experimentA.py

Outputs (in resultats/resultats-{scenario}/)
------------------------------------------------
    paired_diff_n5.csv / paired_diff_n20.csv   (RL vs each baseline, table)
    fig3b_paired_diff.png                      (forest-style plot, n=20, centered at 0)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_ROOT = "resultats"
SCENARIOS    = ["urban_light", "urban_medium", "urban_dense"]
WARMUP       = 50

REFERENCE = "trained"
BASELINES = ["no_cr", "random", "static", "exhaustive_greedy"]

STRATEGY_LABELS = {
    "no_cr":             "No-CR",
    "random":            "Random",
    "static":            "Static",
    "exhaustive_greedy": "Greedy (oracle)",
    "trained":           "RL Agent",
}
SCENARIO_LABELS = {
    "urban_light":  "Urban-Light",
    "urban_medium": "Urban-Medium",
    "urban_dense":  "Urban-Dense",
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 100})


def _eval_dir(scenario):
    return os.path.join(RESULTS_ROOT, f"resultats-{scenario}", "logs", "eval")


def _seed_mean(scenario, strategy_key, seed):
    path = os.path.join(_eval_dir(scenario), strategy_key, f"s{seed}", "satisfaction_summary.csv")
    df = pd.read_csv(path)
    df = df[df["Step"] > WARMUP]
    return float(df["Rate_Global"].mean())


def _ci95(values):
    n = len(values)
    return 0.0 if n < 2 else 1.96 * float(np.std(values, ddof=1)) / np.sqrt(n)


def paired_diffs(scenario, seeds):
    rows = []
    for baseline in BASELINES:
        diffs = [_seed_mean(scenario, REFERENCE, s) - _seed_mean(scenario, baseline, s) for s in seeds]
        mean_diff = float(np.mean(diffs))
        ci = _ci95(diffs)
        rows.append({
            "comparison": f"RL Agent - {STRATEGY_LABELS[baseline]}",
            "mean_diff": round(mean_diff, 4),
            "std_diff": round(float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0, 4),
            "ci95": round(ci, 4),
            "n": len(seeds),
            "significant_at_95": bool(abs(mean_diff) > ci),
        })
    return rows


def make_plot(rows, scenario, n_seeds, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        color = "#27ae60" if r["significant_at_95"] else "#7f8c8d"
        ax.errorbar(r["mean_diff"], yi, xerr=r["ci95"], fmt="o", color=color,
                    markersize=9, capsize=6, elinewidth=2.5, capthick=2.5, zorder=3)
    ax.axvline(0, color="black", lw=1.2, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["comparison"] for r in rows], fontsize=11)
    ax.set_xlabel("Paired difference in mean satisfaction rate (RL - baseline)")
    ax.set_title(
        f"Paired seed-by-seed differences -- {SCENARIO_LABELS[scenario]}\n"
        f"Mean +/- 95% CI on paired diff  (n = {n_seeds} seeds)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(scenario, n_seeds, make_fig):
    seeds = list(range(n_seeds))
    rows = paired_diffs(scenario, seeds)
    df = pd.DataFrame(rows)

    out_dir = os.path.join(RESULTS_ROOT, f"resultats-{scenario}")
    csv_path = os.path.join(out_dir, f"paired_diff_n{n_seeds}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n=== {scenario}  n={n_seeds} ===")
    print(df.to_string(index=False))
    print(f"Saved: {csv_path}")

    if make_fig:
        fig_path = os.path.join(out_dir, "fig3b_paired_diff.png")
        make_plot(rows, scenario, n_seeds, fig_path)
        print(f"Saved: {fig_path}")


if __name__ == "__main__":
    for sc in SCENARIOS:
        run(sc, 5, make_fig=False)
        run(sc, 20, make_fig=True)
