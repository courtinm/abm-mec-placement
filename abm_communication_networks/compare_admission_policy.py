"""
compare_admission_policy.py
------------------------------
Before/after comparison of the CR admission policy fix (dissertation
Section 5.3.1): FCFS (list-order) admission vs.\ application-priority
admission (AR/VR > streaming > best-effort, ties broken by user.id),
trained RL placement agent only, across all three scenarios.

Reuses the RN/CR Q-tables already trained by generate_results.py
(resultats/resultats-{scenario}/models/). Does NOT retrain anything -- if
those Q-table files are missing, the script stops with an error message
telling you to train first.

Both policies are simulated from an otherwise identical run (same seed,
same scenario, same trained agent, same call to sim.simulate_step()): seed
i produces identical user trajectories under both policies (the paired
design of Section 6.3.3), so the comparison is reported both as
independent means +/- 95% CI and as a paired seed-by-seed difference
(priority - fcfs) per application type, the latter being the statistically
appropriate test given the shared-seed design (see the paired-difference
methodology used throughout Chapter 7).

Each (scenario, policy, seed) eval run is cached under OUTPUT_ROOT: if its
satisfaction_summary.csv already exists, it is reused instead of
re-simulated, so re-running after an interruption only simulates what is
missing.

Usage
-----
    python compare_admission_policy.py                        # all 3 scenarios, 20 seeds, 300 steps
    python compare_admission_policy.py --seeds 1 --steps 60    # quick smoke test
    python compare_admission_policy.py --scenarios urban_medium --seeds 5

Outputs (in admission_policy_comparison/)
------------------------------------------
    eval/{scenario}/{policy}/s{seed}/satisfaction_summary.csv  (per-run logs, cached)
    admission_policy_comparison.csv       (aggregated per-app table, mean +/- CI95)
    admission_policy_paired_diff.csv      (paired per-app diff, priority - fcfs)
    admission_policy_comparison.png       (grouped bar chart, one panel per scenario)
"""

import argparse
import importlib
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from main import build_simulation
from agents.cr_placement_agent import CRPlacementAgent

SCENARIOS_DEFAULT = ["urban_light", "urban_medium", "urban_dense"]
POLICIES    = ["fcfs", "priority"]
APP_COLS    = ["Rate_AR_VR", "Rate_Streaming", "Rate_BestEffort"]
APP_LABELS  = {
    "Rate_AR_VR":      "AR/VR",
    "Rate_Streaming":  "Streaming",
    "Rate_BestEffort": "Best-effort",
}
WARMUP      = 50
EVAL_STEPS  = 300

OUTPUT_ROOT = "admission_policy_comparison"
EVAL_DIR    = os.path.join(OUTPUT_ROOT, "eval")

POLICY_LABELS = {"fcfs": "FCFS (before)", "priority": "App-priority (after)"}
POLICY_COLORS = {"fcfs": "#e74c3c", "priority": "#2980b9"}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 100})


def _rn_model_path(models_dir, rn_id):
    return os.path.join(models_dir, f"rn_{rn_id}.pkl")


def _check_models_exist(scenario, models_dir, config):
    missing = []
    for i in range(len(config["relay_nodes"])):
        p = _rn_model_path(models_dir, i + 1)
        if not os.path.exists(p):
            missing.append(p)
    cr_path = os.path.join(models_dir, "cr_qtable.pkl")
    if not os.path.exists(cr_path):
        missing.append(cr_path)
    if missing:
        raise SystemExit(
            f"Missing trained Q-table(s) for {scenario}:\n  " + "\n  ".join(missing) +
            f"\nTrain them first, e.g.: python generate_results.py --scenarios {scenario}"
        )


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def _ci95(values):
    n = len(values)
    return 0.0 if n < 2 else 1.96 * float(np.std(values, ddof=1)) / np.sqrt(n)


def run_eval(scenario, policy, seed, steps, models_dir, out_dir):
    _seed_all(seed)
    config = importlib.import_module(f"configs.{scenario}").CONFIG
    sim = build_simulation(config)
    sim.dynamic_rn = False

    for rn in sim.relay_nodes:
        path = _rn_model_path(models_dir, rn.id)
        if os.path.exists(path):
            rn.agent.load_qtable(path)
            rn.agent.frozen = True
            rn.agent.epsilon = 0.0

    for bs in sim.base_stations:
        bs.has_compute_resource = False
        bs.compute_resource = None

    cr_cfg = config.get("cr_placement", {})
    agent = CRPlacementAgent(
        sim.base_stations,
        cr_cfg.get("k", 2),
        cr_cfg.get("cr_capacity_mbps", 100.0),
    )
    agent._users = sim.users
    agent._relay_nodes = sim.relay_nodes
    cr_path = os.path.join(models_dir, "cr_qtable.pkl")
    agent.load_qtable(cr_path)
    agent.frozen = True
    agent.epsilon = 0.0
    sim.cr_agent = agent

    sim.metrics.admission_policy = policy  # "fcfs" or "priority" — see simulator.py

    for _ in range(steps):
        sim.simulate_step()
    sim.finalize(output_dir=out_dir)


def ensure_run(scenario, policy, seed, steps, models_dir):
    """Run eval for (scenario, policy, seed) unless its output is already cached."""
    out_dir = os.path.join(EVAL_DIR, scenario, policy, f"s{seed}")
    csv_path = os.path.join(out_dir, "satisfaction_summary.csv")
    if os.path.exists(csv_path):
        return "cached"
    os.makedirs(out_dir, exist_ok=True)
    run_eval(scenario, policy, seed, steps, models_dir, out_dir)
    return "simulated"


def _per_app_means(scenario, policy, seeds):
    """Per-seed post-warmup mean satisfaction, one list per app column."""
    per_app = {c: [] for c in APP_COLS}
    for seed in seeds:
        path = os.path.join(EVAL_DIR, scenario, policy, f"s{seed}", "satisfaction_summary.csv")
        df = pd.read_csv(path)
        df = df[df["Step"] > WARMUP]
        for c in APP_COLS:
            per_app[c].append(float(df[c].mean()))
    return per_app


def make_figure(summary_df, scenarios, out_path):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4.5), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    app_order = [APP_LABELS[c] for c in APP_COLS]

    for ax, scenario in zip(axes, scenarios):
        sub = summary_df[summary_df["scenario"] == scenario]
        x = np.arange(len(app_order))
        width = 0.35
        for i, policy in enumerate(POLICIES):
            rows = sub[sub["policy"] == policy].set_index("app").loc[app_order]
            ax.bar(x + (i - 0.5) * width, rows["mean"], width, yerr=rows["ci95"],
                   capsize=5, label=POLICY_LABELS[policy], color=POLICY_COLORS[policy],
                   alpha=0.87, edgecolor="white", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(app_order, fontsize=10)
        ax.set_title(scenario, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel("Mean satisfaction rate (post-warmup)")
    axes[-1].legend(fontsize=10, loc="upper right")
    fig.suptitle("CR admission policy: FCFS vs. application-priority — trained RL agent",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS_DEFAULT)
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds (default: 20)")
    parser.add_argument("--steps", type=int, default=EVAL_STEPS, help="Eval steps per run (default: 300)")
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    summary_rows = []
    paired_rows = []

    for scenario in args.scenarios:
        models_dir = os.path.join("resultats", f"resultats-{scenario}", "models")
        config = importlib.import_module(f"configs.{scenario}").CONFIG
        _check_models_exist(scenario, models_dir, config)

        print(f"\n{'=' * 62}\n  SCENARIO : {scenario.upper()}\n{'=' * 62}")
        for policy in POLICIES:
            print(f"\n  [{POLICY_LABELS[policy]}]")
            for seed in seeds:
                status = ensure_run(scenario, policy, seed, args.steps, models_dir)
                print(f"    seed {seed}: {status}")

        per_app = {policy: _per_app_means(scenario, policy, seeds) for policy in POLICIES}

        for app_col in APP_COLS:
            for policy in POLICIES:
                vals = per_app[policy][app_col]
                summary_rows.append({
                    "scenario": scenario,
                    "policy": policy,
                    "app": APP_LABELS[app_col],
                    "mean": round(float(np.mean(vals)), 4),
                    "ci95": round(_ci95(vals), 4),
                    "n": len(vals),
                })
            diffs = [p - f for p, f in zip(per_app["priority"][app_col], per_app["fcfs"][app_col])]
            mean_diff = float(np.mean(diffs))
            ci = _ci95(diffs)
            paired_rows.append({
                "scenario": scenario,
                "app": APP_LABELS[app_col],
                "mean_diff": round(mean_diff, 4),
                "ci95": round(ci, 4),
                "n": len(diffs),
                "significant_at_95": bool(abs(mean_diff) > ci),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_ROOT, "admission_policy_comparison.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved: {summary_csv}")

    paired_df = pd.DataFrame(paired_rows)
    paired_csv = os.path.join(OUTPUT_ROOT, "admission_policy_paired_diff.csv")
    paired_df.to_csv(paired_csv, index=False)
    print(f"Saved: {paired_csv}")
    print("\n" + paired_df.to_string(index=False))

    fig_path = os.path.join(OUTPUT_ROOT, "admission_policy_comparison.png")
    make_figure(summary_df, args.scenarios, fig_path)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
