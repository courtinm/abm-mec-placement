"""
analyze_experiment_B.py
------------------------
Experiment B — trained RL agent across scenarios (urban_light -> urban_medium
-> urban_dense). Increasing user density and AR/VR share tightens latency
constraints; this script relates CR saturation (rho = load / capacity) to the
resulting satisfaction degradation.

Reuses the eval/trained/{scenario}/s{seed}/ logs already produced by
generate_results.py — no new simulation runs are needed.

Usage
-----
    python analyze_experiment_B.py

Outputs (in resultats/)
------------------------
    expB_satisfaction_progression.png
    expB_cr_utilization.png
    expB_summary.csv
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_ROOT = "resultats"
SCENARIOS    = ["urban_light", "urban_medium", "urban_dense"]
SEEDS        = list(range(20))
WARMUP       = 50
DPI          = 300

SCENARIO_LABELS = {
    "urban_light":  "Urban-Light\n(20 users)",
    "urban_medium": "Urban-Medium\n(30 users)",
    "urban_dense":  "Urban-Dense\n(40 users)",
}
SCENARIO_COLORS = {
    "urban_light":  "#2ecc71",
    "urban_medium": "#3498db",
    "urban_dense":  "#e67e22",
}
APP_COLS = ["Rate_Global", "Rate_AR_VR", "Rate_Streaming", "Rate_BestEffort"]
APP_LABELS = {
    "Rate_Global":      "Global",
    "Rate_AR_VR":       "AR/VR",
    "Rate_Streaming":   "Streaming",
    "Rate_BestEffort":  "Best-effort",
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 100})


def _ci95(values):
    n = len(values)
    return 0.0 if n < 2 else 1.96 * float(np.std(values, ddof=1)) / np.sqrt(n)


def _eval_dir(scenario, seed):
    return os.path.join(RESULTS_ROOT, f"resultats-{scenario}", "logs", "eval", "trained", f"s{seed}")


def _collect_satisfaction(scenario):
    """Per-seed post-warmup means for each app column."""
    data = {c: [] for c in APP_COLS}
    for seed in SEEDS:
        path = os.path.join(_eval_dir(scenario, seed), "satisfaction_summary.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df["Step"] > WARMUP]
        if df.empty:
            continue
        for c in APP_COLS:
            data[c].append(float(df[c].mean()))
    return data


def _collect_cr_utilization(scenario):
    """Per-seed: mean rho and overflow rate (share of BS-steps with rho > 1)."""
    mean_rho, overflow_rate = [], []
    for seed in SEEDS:
        path = os.path.join(_eval_dir(scenario, seed), "cr_utilization.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df["Step"] > WARMUP]
        if df.empty:
            continue
        mean_rho.append(float(df["CR_utilization"].mean()))
        overflow_rate.append(float((df["CR_utilization"] > 1.0).mean()))
    return mean_rho, overflow_rate


def fig_satisfaction_progression(all_sat, out_dir):
    n_apps = len(APP_COLS)
    n_sc = len(SCENARIOS)
    group_w = 0.75
    bar_w = group_w / n_sc
    x = np.arange(n_apps)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, sc in enumerate(SCENARIOS):
        data = all_sat[sc]
        offsets = x + (idx - n_sc / 2 + 0.5) * bar_w
        m = [np.mean(data[c]) if data[c] else 0.0 for c in APP_COLS]
        ci = [_ci95(data[c]) for c in APP_COLS]
        ax.bar(offsets, m, yerr=ci, width=bar_w * 0.88,
               color=SCENARIO_COLORS[sc], alpha=0.87, edgecolor="white",
               linewidth=1, capsize=5, label=SCENARIO_LABELS[sc],
               error_kw={"elinewidth": 1.5, "ecolor": "#333333"})

    ax.set_xticks(x)
    ax.set_xticklabels([APP_LABELS[c] for c in APP_COLS], fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Mean satisfaction rate (post-warmup)")
    ax.set_title("Experiment B — Satisfaction vs scenario load (RL agent)",
                 fontweight="bold")
    ax.axhline(1.0, color="grey", linestyle="--", lw=1, alpha=0.5)
    ax.legend(title="Scenario", fontsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "expB_satisfaction_progression.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def fig_cr_utilization(all_rho, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: mean rho per scenario
    ax = axes[0]
    means = [np.mean(all_rho[sc][0]) if all_rho[sc][0] else 0.0 for sc in SCENARIOS]
    cis   = [_ci95(all_rho[sc][0]) for sc in SCENARIOS]
    colors = [SCENARIO_COLORS[sc] for sc in SCENARIOS]
    x = np.arange(len(SCENARIOS))
    ax.bar(x, means, yerr=cis, capsize=6, color=colors, alpha=0.87,
           edgecolor="white", linewidth=1.2,
           error_kw={"elinewidth": 2, "ecolor": "#333333"})
    ax.axhline(1.0, color="crimson", linestyle="--", lw=1.5, label="rho = 1 (saturation)")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[sc] for sc in SCENARIOS], fontsize=10)
    ax.set_ylabel(r"Mean CR utilisation  $\rho = \lambda / C$")
    ax.set_title("Mean CR load factor", fontweight="bold")
    ax.legend(fontsize=9)

    # Right: overflow rate per scenario
    ax = axes[1]
    means = [np.mean(all_rho[sc][1]) if all_rho[sc][1] else 0.0 for sc in SCENARIOS]
    cis   = [_ci95(all_rho[sc][1]) for sc in SCENARIOS]
    ax.bar(x, means, yerr=cis, capsize=6, color=colors, alpha=0.87,
           edgecolor="white", linewidth=1.2,
           error_kw={"elinewidth": 2, "ecolor": "#333333"})
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[sc] for sc in SCENARIOS], fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Share of BS-steps with rho > 1")
    ax.set_title("CR overflow frequency", fontweight="bold")

    fig.suptitle("Experiment B — CR saturation across scenarios", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "expB_cr_utilization.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def write_summary_csv(all_sat, all_rho, out_dir):
    path = os.path.join(out_dir, "expB_summary.csv")
    rows = []
    for sc in SCENARIOS:
        row = {"scenario": sc}
        for c in APP_COLS:
            vals = all_sat[sc][c]
            row[f"{c}_mean"] = round(np.mean(vals), 4) if vals else None
            row[f"{c}_ci95"] = round(_ci95(vals), 4) if vals else None
        rho_vals, overflow_vals = all_rho[sc]
        row["rho_mean"] = round(np.mean(rho_vals), 4) if rho_vals else None
        row["rho_ci95"] = round(_ci95(rho_vals), 4) if rho_vals else None
        row["overflow_rate_mean"] = round(np.mean(overflow_vals), 4) if overflow_vals else None
        row["overflow_rate_ci95"] = round(_ci95(overflow_vals), 4) if overflow_vals else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  -> {path}")


def main():
    print("Experiment B — RL agent across scenarios (urban_light -> medium -> dense)")
    all_sat = {}
    all_rho = {}
    for sc in SCENARIOS:
        all_sat[sc] = _collect_satisfaction(sc)
        all_rho[sc] = _collect_cr_utilization(sc)
        n_seeds = len(all_sat[sc]["Rate_Global"])
        if n_seeds < len(SEEDS):
            print(f"  [WARNING] {sc}: only {n_seeds}/{len(SEEDS)} seeds found.")

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    fig_satisfaction_progression(all_sat, RESULTS_ROOT)
    fig_cr_utilization(all_rho, RESULTS_ROOT)
    write_summary_csv(all_sat, all_rho, RESULTS_ROOT)
    print("Done.")


if __name__ == "__main__":
    main()
