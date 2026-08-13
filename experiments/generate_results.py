"""
generate_results.py
-------------------
Runs all experiments (train + eval) and generates the figures for each scenario.

Usage
-----
    python experiments/generate_results.py                         # all scenarios, default parameters
    python experiments/generate_results.py --scenarios urban       # a single scenario
    python experiments/generate_results.py --steps-train 200       # fast mode for testing

(always run from the project root, abm_communication_networks/)

Output structure
-----------------
    output/
      output-urban_light/
        models/   rn_1.pkl, cr_qtable.pkl
        logs/     train_rn/rn_reward.csv
                  train_cr/cr_reward.csv, cr_delta_q.csv
                  eval/trained/s{seed}/...
                  eval/{strategy}/s{seed}/...
        fig1_training_curves.png
        fig2_delta_q.png
        fig3_strategy_comparison.png
        fig4_per_app_satisfaction.png
        fig5_satisfaction_timeseries.png
      output-urban_medium/
      output-urban_dense/
"""

import argparse
import csv
import importlib
import os
import random
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# Allow `python experiments/generate_results.py` to find the project-root
# packages (main.py, agents/, configs/) regardless of the current directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import build_simulation
from agents.cr_placement_agent import CRPlacementAgent
from agents.placement_strategies import make_strategy

# ── Global parameters ──────────────────────────────────────────────────────────

SCENARIOS_DEFAULT = ["urban_light", "urban_medium", "urban_dense"]
STRATEGIES        = ["no_cr", "random", "static", "exhaustive_greedy"]
SEEDS_DEFAULT     = [0, 1, 2, 3, 4]

TRAIN_RN_STEPS = 1000
TRAIN_CR_STEPS = {
    "urban_light":  5000,
    "urban_medium": 2000,
    "urban_dense":  3000,
}
EVAL_STEPS     = 300
WARMUP         = 50
MA_WINDOW      = 30   # moving-average window for the figures

# 2x2 factorial design (dissertation Ch.7, M0-M3). M0 = both at their
# simulator defaults ("fcfs", "equal_share") -> label None, meaning the
# ORIGINAL output/output-{scenario}/ path is used unchanged, so M0's
# already-trained models/results are never touched by this factorial support
# unless explicitly targeted (see --skip-train guard in main()).
CONDITION_LABELS = {
    ("fcfs",     "equal_share"):  None,   # M0 — do not retrain (existing results)
    ("priority", "equal_share"):  "M1",
    ("fcfs",     "proportional"): "M2",
    ("priority", "proportional"): "M3",
}
DPI            = 300

STRATEGY_COLORS = {
    "no_cr":             "#95a5a6",
    "random":            "#e74c3c",
    "static":            "#f39c12",
    "exhaustive_greedy": "#27ae60",
    "trained":           "#2980b9",
}
STRATEGY_LABELS = {
    "no_cr":             "No CR",
    "random":            "Random",
    "static":            "Static (Macro BSs)",
    "exhaustive_greedy": "Greedy (oracle)",
    "trained":           "RL Agent (Q-learning)",
}
APP_COLS = ["Rate_AR_VR", "Rate_Streaming", "Rate_BestEffort"]
APP_LABELS = {
    "Rate_AR_VR":       "AR/VR\n(10 ms / 25 Mbps)",
    "Rate_Streaming":   "Streaming\n(50 ms / 10 Mbps)",
    "Rate_BestEffort":  "Best-effort\n(200 ms / 1 Mbps)",
}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 100,
})
plt.style.use("seaborn-v0_8-whitegrid")


# ── Utilities ──────────────────────────────────────────────────────────────────

def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def _load_config(scenario):
    return importlib.import_module(f"configs.{scenario}").CONFIG


def _rn_model_path(models_dir, rn_id):
    return os.path.join(models_dir, f"rn_{rn_id}.pkl")


def _moving_avg(series, w):
    return pd.Series(series).rolling(w, center=True, min_periods=1).mean().to_numpy()


def _ci95(values):
    n = len(values)
    return 0.0 if n < 2 else 1.96 * float(np.std(values, ddof=1)) / np.sqrt(n)


def _collect_satisfaction(eval_base, strategy, warmup):
    """Returns the list of mean satisfaction rates (post-warmup), one per seed."""
    results = []
    strat_dir = os.path.join(eval_base, strategy)
    if not os.path.isdir(strat_dir):
        return results
    for seed_dir in sorted(os.listdir(strat_dir)):
        path = os.path.join(strat_dir, seed_dir, "satisfaction_summary.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df["Step"] > warmup]
        if not df.empty:
            results.append(float(df["Rate_Global"].mean()))
    return results


def _collect_timeseries(eval_base, strategy, warmup):
    """Returns a step→mean_rate Series averaged over all seeds (post-warmup)."""
    strat_dir = os.path.join(eval_base, strategy)
    if not os.path.isdir(strat_dir):
        return None
    all_dfs = []
    for seed_dir in sorted(os.listdir(strat_dir)):
        path = os.path.join(strat_dir, seed_dir, "satisfaction_summary.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df["Step"] > warmup][["Step", "Rate_Global"]].set_index("Step")
        all_dfs.append(df)
    if not all_dfs:
        return None
    combined = pd.concat(all_dfs, axis=1)
    combined.columns = range(len(combined.columns))
    return combined.mean(axis=1)


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ── Experiment phases ─────────────────────────────────────────────────────────

def run_train_rn(scenario, steps, models_dir, logs_dir,
                  cr_admission_policy="fcfs", radio_allocation="equal_share"):
    """Phase 1 — RN (positioning) agent training."""
    print(f"\n  [train_rn] {scenario}  ({steps} steps)", flush=True)
    _seed_all(42)
    config = _load_config(scenario)
    sim = build_simulation(config)
    sim.dynamic_rn = False
    sim.cr_admission_policy = cr_admission_policy  # 2x2 factorial, Ch.7 M0-M3
    sim.radio_allocation    = radio_allocation

    if not sim.relay_nodes:
        print("    -> No RN in this scenario, phase skipped.")
        return

    rn_rows = []
    for _ in range(steps):
        sim.simulate_step()
        rn_rows.append((sim.timestep, sum(rn.last_reward for rn in sim.relay_nodes)))

    os.makedirs(os.path.join(logs_dir, "train_rn"), exist_ok=True)
    _write_csv(os.path.join(logs_dir, "train_rn", "rn_reward.csv"),
               ["Step", "TotalReward"], rn_rows)

    os.makedirs(models_dir, exist_ok=True)
    for rn in sim.relay_nodes:
        rn.agent.save_qtable(_rn_model_path(models_dir, rn.id))

    first20 = [r for _, r in rn_rows[: len(rn_rows) // 5]]
    last20  = [r for _, r in rn_rows[-len(rn_rows) // 5 :]]
    trend   = "↑ improving" if np.mean(last20) > np.mean(first20) else "→ stable"
    print(f"    reward  first 20% = {np.mean(first20):.2f}  "
          f"last 20% = {np.mean(last20):.2f}  {trend}")


def run_train_cr(scenario, steps, models_dir, logs_dir, warmup,
                  cr_admission_policy="fcfs", radio_allocation="equal_share"):
    """Phase 2 — CR agent training (RN frozen)."""
    print(f"\n  [train_cr] {scenario}  ({steps} steps)", flush=True)
    _seed_all(42)
    config = _load_config(scenario)
    sim = build_simulation(config)
    sim.dynamic_rn = False
    sim.cr_admission_policy = cr_admission_policy  # 2x2 factorial, Ch.7 M0-M3
    sim.radio_allocation    = radio_allocation

    for rn in sim.relay_nodes:
        path = _rn_model_path(models_dir, rn.id)
        if os.path.exists(path):
            rn.agent.load_qtable(path)
            rn.agent.frozen = True
            rn.agent.epsilon = 0.0

    # Reset the CRs defined in the config before attaching the agent
    for bs in sim.base_stations:
        bs.has_compute_resource = False
        bs.compute_resource = None

    cr_cfg = config.get("cr_placement", {})
    hp = cr_cfg.get("rl_hyperparams", {})
    agent = CRPlacementAgent(
        sim.base_stations,
        cr_cfg.get("k", 2),
        cr_cfg.get("cr_capacity_mbps", 100.0),
        epsilon=hp.get("epsilon_0", 0.5),
        epsilon_min=hp.get("epsilon_min", 0.05),
        epsilon_decay=hp.get("epsilon_decay", 0.995),
        alpha_min=hp.get("alpha_min", 0.01),
        alpha_decay=hp.get("alpha_decay", 0.998),
        reward_shaping_lambda=hp.get("reward_shaping_lambda", 0.0),
    )
    agent._users = sim.users
    agent._relay_nodes = sim.relay_nodes
    sim.cr_agent = agent

    cr_rows = []
    cr_global_rows = []
    for _ in range(steps):
        sim.simulate_step()
        if sim.timestep > warmup and agent.last_reward is not None:
            cr_rows.append((sim.timestep, round(agent.last_reward, 4)))
            if agent.last_reward_global is not None:
                cr_global_rows.append((sim.timestep, round(agent.last_reward_global, 4)))

    os.makedirs(os.path.join(logs_dir, "train_cr"), exist_ok=True)
    _write_csv(os.path.join(logs_dir, "train_cr", "cr_reward.csv"),
               ["Step", "SatisfactionRate"], cr_rows)
    _write_csv(os.path.join(logs_dir, "train_cr", "cr_reward_global.csv"),
               ["Step", "SatisfactionRate"], cr_global_rows)
    _write_csv(os.path.join(logs_dir, "train_cr", "cr_delta_q.csv"),
               ["Step", "DeltaQ"],
               list(enumerate(agent.delta_q_history, 1)))

    os.makedirs(models_dir, exist_ok=True)
    agent.save_qtable(os.path.join(models_dir, "cr_qtable.pkl"))

    if cr_rows:
        vals = [r for _, r in cr_rows]
        first20 = vals[: len(vals) // 5]
        last20  = vals[-len(vals) // 5 :]
        trend   = "↑ improving" if np.mean(last20) > np.mean(first20) else "→ stable"
        print(f"    satisfaction  first 20% = {np.mean(first20):.3f}  "
              f"last 20% = {np.mean(last20):.3f}  {trend}")


def run_eval_trained(scenario, steps, seed, models_dir, logs_dir,
                      cr_admission_policy="fcfs", radio_allocation="equal_share"):
    """Phase 3 — RL agent evaluation (RN + CR frozen)."""
    out_dir = os.path.join(logs_dir, "eval", "trained", f"s{seed}")
    os.makedirs(out_dir, exist_ok=True)
    _seed_all(seed)
    config = _load_config(scenario)
    sim = build_simulation(config)
    sim.dynamic_rn = False
    sim.cr_admission_policy = cr_admission_policy  # 2x2 factorial, Ch.7 M0-M3
    sim.radio_allocation    = radio_allocation

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
    if os.path.exists(cr_path):
        agent.load_qtable(cr_path)
    agent.frozen = True
    agent.epsilon = 0.0
    sim.cr_agent = agent

    for _ in range(steps):
        sim.simulate_step()
    sim.finalize(output_dir=out_dir)


def run_eval_strategy(scenario, strategy_name, steps, seed, models_dir, logs_dir,
                       cr_admission_policy="fcfs", radio_allocation="equal_share"):
    """Phase 4 — baseline strategy evaluation (RN frozen, no CR Q-learning)."""
    out_dir = os.path.join(logs_dir, "eval", strategy_name, f"s{seed}")
    os.makedirs(out_dir, exist_ok=True)
    _seed_all(seed)
    config = _load_config(scenario)
    sim = build_simulation(config)
    sim.dynamic_rn = False
    sim.cr_admission_policy = cr_admission_policy  # 2x2 factorial, Ch.7 M0-M3
    sim.radio_allocation    = radio_allocation

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
    strategy = make_strategy(
        strategy_name,
        sim.base_stations,
        k=cr_cfg.get("k", 2),
        cr_capacity_mbps=cr_cfg.get("cr_capacity_mbps", 100.0),
    )
    if hasattr(strategy, "_users"):
        strategy._users = sim.users
    if hasattr(strategy, "radio_allocation"):
        strategy.radio_allocation = sim.radio_allocation
    if hasattr(strategy, "cr_admission_policy"):
        strategy.cr_admission_policy = sim.cr_admission_policy
    sim.cr_agent = strategy

    for _ in range(steps):
        sim.simulate_step()
    sim.finalize(output_dir=out_dir)


# ── Figures ────────────────────────────────────────────────────────────────────

def _fig1_training_curves(logs_dir, out_dir):
    rn_path = os.path.join(logs_dir, "train_rn", "rn_reward.csv")
    cr_path = os.path.join(logs_dir, "train_cr", "cr_reward.csv")
    rn_df = pd.read_csv(rn_path) if os.path.exists(rn_path) else None
    cr_df = pd.read_csv(cr_path) if os.path.exists(cr_path) else None

    if rn_df is None and cr_df is None:
        print("    [SKIP] fig1 — no training data")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle("Reinforcement Learning Training Convergence",
                 fontweight="bold", y=1.01, fontsize=14)

    # Subplot 1 : RN
    ax = axes[0]
    if rn_df is not None:
        x, y = rn_df["Step"].to_numpy(), rn_df["TotalReward"].to_numpy()
        ax.plot(x, y, color="#d0e4f0", lw=1, label="Raw")
        ax.plot(x, _moving_avg(y, MA_WINDOW), color="steelblue", lw=2,
                label=f"Moving average (w = {MA_WINDOW})")
        ax.set_ylabel("Total reward\n(users connected to RN)")
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "rn_reward.csv not found", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    ax.set_title("RN Agent — position learning (Q-learning)")
    ax.set_xlabel("Training step")

    # Subplot 2 : CR
    ax = axes[1]
    if cr_df is not None:
        x, y = cr_df["Step"].to_numpy(), cr_df["SatisfactionRate"].to_numpy()
        ax.plot(x, y, color="#fde8c8", lw=1, label="Raw")
        ax.plot(x, _moving_avg(y, MA_WINDOW), color="darkorange", lw=2,
                label=f"Moving average (w = {MA_WINDOW})")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylabel("Global satisfaction rate")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "cr_reward.csv not found", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    ax.set_title("CR Agent — compute resource placement learning (Q-learning)")
    ax.set_xlabel("Training step")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_training_curves.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


def _fig2_delta_q(logs_dir, out_dir):
    dq_path = os.path.join(logs_dir, "train_cr", "cr_delta_q.csv")
    if not os.path.exists(dq_path):
        print("    [SKIP] fig2 — cr_delta_q.csv not found")
        return

    df = pd.read_csv(dq_path)
    x, y = df["Step"].to_numpy(), df["DeltaQ"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color="#f0cccc", lw=1, label="Raw |ΔQ|")
    ax.plot(x, _moving_avg(y, MA_WINDOW), color="crimson", lw=2,
            label=f"Moving average (w = {MA_WINDOW})")
    ax.axhline(0.01, color="navy", linestyle="--", lw=1.5,
               label="Convergence threshold (ΔQ = 0.01)")
    ax.set_yscale("log")
    ax.set_xlabel("Q-update")
    ax.set_ylabel("|ΔQ|  (log scale)")
    ax.set_title("Q-value Convergence — CR Placement Agent",
                 fontweight="bold")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_delta_q.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


def _fig3_strategy_comparison(logs_dir, out_dir, scenario, warmup):
    eval_dir = os.path.join(logs_dir, "eval")
    all_keys = STRATEGIES + ["trained"]
    means, cis, valid = [], [], []

    n_seeds = 0
    for key in all_keys:
        seeds = _collect_satisfaction(eval_dir, key, warmup)
        if not seeds:
            continue
        means.append(np.mean(seeds))
        cis.append(_ci95(seeds))
        valid.append(key)
        n_seeds = max(n_seeds, len(seeds))

    if not valid:
        print("    [SKIP] fig3 — no eval data")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(valid))
    colors   = [STRATEGY_COLORS[k] for k in valid]
    xlabels  = [STRATEGY_LABELS[k] for k in valid]

    bars = ax.bar(x, means, yerr=cis, capsize=7, color=colors, alpha=0.87,
                  edgecolor="white", linewidth=1.2,
                  error_kw={"elinewidth": 2, "ecolor": "#333333", "capthick": 2})

    for bar, m, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, m + ci + 0.018,
                f"{m:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(1.0, color="grey", linestyle="--", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Mean satisfaction rate (post-warmup)")
    ax.set_title(
        f"Strategy comparison — {scenario} scenario\n"
        f"Global satisfaction ± 95% CI  (n = {n_seeds} seeds)",
        fontweight="bold",
    )

    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_strategy_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


def _fig4_per_app_satisfaction(logs_dir, out_dir, warmup):
    eval_dir  = os.path.join(logs_dir, "eval")
    all_keys  = STRATEGIES + ["trained"]

    data = {}
    for key in all_keys:
        strat_dir = os.path.join(eval_dir, key)
        if not os.path.isdir(strat_dir):
            continue
        col_data = {c: [] for c in APP_COLS}
        for seed_dir in sorted(os.listdir(strat_dir)):
            path = os.path.join(strat_dir, seed_dir, "satisfaction_summary.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df = df[df["Step"] > warmup]
            if df.empty:
                continue
            for col in APP_COLS:
                if col in df.columns:
                    col_data[col].append(float(df[col].mean()))
        if any(col_data[c] for c in APP_COLS):
            data[key] = col_data

    if not data:
        print("    [SKIP] fig4 — no data")
        return

    n_apps  = len(APP_COLS)
    n_strat = len(data)
    group_w = 0.8
    bar_w   = group_w / n_strat
    x       = np.arange(n_apps)

    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, (key, col_data) in enumerate(data.items()):
        offsets = x + (idx - n_strat / 2 + 0.5) * bar_w
        m  = [np.mean(col_data[c]) if col_data[c] else 0.0 for c in APP_COLS]
        ci = [_ci95(col_data[c]) for c in APP_COLS]
        ax.bar(offsets, m, yerr=ci, width=bar_w * 0.90,
               color=STRATEGY_COLORS[key], alpha=0.87, edgecolor="white",
               linewidth=1, capsize=4, label=STRATEGY_LABELS[key],
               error_kw={"elinewidth": 1.5, "ecolor": "#333333"})

    ax.set_xticks(x)
    ax.set_xticklabels([APP_LABELS[c] for c in APP_COLS], fontsize=11)
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Mean satisfaction rate")
    ax.set_title("Per-application satisfaction by strategy",
                 fontweight="bold")
    ax.axhline(1.0, color="grey", linestyle="--", lw=1, alpha=0.5)
    ax.legend(title="Strategy", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_per_app_satisfaction.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


def _fig5_satisfaction_timeseries(logs_dir, out_dir, warmup):
    eval_dir = os.path.join(logs_dir, "eval")
    all_keys = STRATEGIES + ["trained"]

    fig, ax = plt.subplots(figsize=(11, 5))
    plotted = False
    for key in all_keys:
        series = _collect_timeseries(eval_dir, key, warmup)
        if series is None or series.empty:
            continue
        steps  = series.index.to_numpy()
        values = series.to_numpy()
        ax.plot(
            steps, values,
            color=STRATEGY_COLORS[key],
            lw=2.5 if key == "trained" else 1.5,
            linestyle="-" if key == "trained" else "--",
            label=STRATEGY_LABELS[key],
            alpha=0.9,
            zorder=3 if key == "trained" else 2,
        )
        plotted = True

    if not plotted:
        print("    [SKIP] fig5 — no eval data")
        plt.close(fig)
        return

    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Evaluation step")
    ax.set_ylabel("Satisfaction rate (mean over seeds)")
    ax.set_title("Satisfaction rate over evaluation steps",
                 fontweight="bold")
    ax.legend(title="Strategy", fontsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_satisfaction_timeseries.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


def generate_all_plots(logs_dir, out_dir, scenario):
    print(f"\n  [plots] {scenario}")
    os.makedirs(out_dir, exist_ok=True)
    _fig1_training_curves(logs_dir, out_dir)
    _fig2_delta_q(logs_dir, out_dir)
    _fig3_strategy_comparison(logs_dir, out_dir, scenario, WARMUP)
    _fig4_per_app_satisfaction(logs_dir, out_dir, WARMUP)
    _fig5_satisfaction_timeseries(logs_dir, out_dir, WARMUP)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Runs all experiments and generates the figures for each scenario."
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=SCENARIOS_DEFAULT,
        help=f"Scenarios to process (default: {SCENARIOS_DEFAULT})"
    )
    parser.add_argument(
        "--steps-train", type=int, default=TRAIN_RN_STEPS,
        help=f"RN training steps (default: {TRAIN_RN_STEPS}). "
             f"CR steps: {TRAIN_CR_STEPS} (per-scenario values, not configurable via CLI)"
    )
    parser.add_argument(
        "--steps-eval", type=int, default=EVAL_STEPS,
        help=f"Evaluation steps per run (default: {EVAL_STEPS})"
    )
    parser.add_argument(
        "--seeds", type=int, default=len(SEEDS_DEFAULT),
        help=f"Number of seeds for evaluation (default: {len(SEEDS_DEFAULT)})"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training (reuse existing models)"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation (regenerate figures only)"
    )
    parser.add_argument(
        "--cr-admission-policy", choices=("fcfs", "priority"), default="fcfs",
        help="2x2 factorial (Ch.7 M0-M3). Default 'fcfs' = M0."
    )
    parser.add_argument(
        "--radio-allocation", choices=("equal_share", "proportional"), default="equal_share",
        help="2x2 factorial (Ch.7 M0-M3). Default 'equal_share' = M0."
    )
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    condition = CONDITION_LABELS[(args.cr_admission_policy, args.radio_allocation)]

    if condition is None and not args.skip_train:
        parser.error(
            "cr-admission-policy=fcfs + radio-allocation=equal_share is M0, whose "
            "Q-tables/results already exist and must not be retrained. Pass "
            "--skip-train if you only want to regenerate M0's plots, or pick a "
            "different --cr-admission-policy/--radio-allocation combination (M1/M2/M3)."
        )

    print("=" * 62)
    print(f"  Condition    : {condition or 'M0 (baseline)'}  "
          f"(cr_admission_policy={args.cr_admission_policy}, radio_allocation={args.radio_allocation})")
    print(f"  Scenarios    : {args.scenarios}")
    print(f"  Steps RN     : {args.steps_train}")
    print(f"  Steps CR     : {TRAIN_CR_STEPS}")
    print(f"  Steps eval   : {args.steps_eval}")
    print(f"  Seeds        : {seeds}")
    print(f"  Skip train   : {args.skip_train}")
    print(f"  Skip eval    : {args.skip_eval}")
    print("=" * 62)

    for scenario in args.scenarios:
        print(f"\n{'='*62}\n  SCENARIO : {scenario.upper()}"
              f"{'  [' + condition + ']' if condition else ''}\n{'='*62}")

        # M0 keeps the original path untouched; M1/M2/M3 get their own
        # distinct, never-colliding directory (see CONDITION_LABELS).
        dir_name   = f"output-{scenario}" + (f"__{condition}" if condition else "")
        base_dir   = os.path.join("output", dir_name)
        logs_dir   = os.path.join(base_dir, "logs")
        models_dir = os.path.join(base_dir, "models")

        # ── Training ──────────────────────────────────────────────────
        if not args.skip_train:
            run_train_rn(scenario, args.steps_train, models_dir, logs_dir,
                        args.cr_admission_policy, args.radio_allocation)
            cr_steps = TRAIN_CR_STEPS.get(scenario, 2000)
            run_train_cr(scenario, cr_steps, models_dir, logs_dir, WARMUP,
                        args.cr_admission_policy, args.radio_allocation)

        # ── Evaluation ────────────────────────────────────────────────
        if not args.skip_eval:
            print(f"\n  [eval] {scenario} — trained RL agent")
            for seed in seeds:
                print(f"    seed {seed} ...", end=" ", flush=True)
                run_eval_trained(scenario, args.steps_eval, seed, models_dir, logs_dir,
                                 args.cr_admission_policy, args.radio_allocation)
                print("ok")

            for strat in STRATEGIES:
                print(f"\n  [eval] {scenario} — baseline '{strat}'")
                for seed in seeds:
                    print(f"    seed {seed} ...", end=" ", flush=True)
                    run_eval_strategy(scenario, strat, args.steps_eval, seed,
                                      models_dir, logs_dir,
                                      args.cr_admission_policy, args.radio_allocation)
                    print("ok")

        # ── Figures ───────────────────────────────────────────────────
        generate_all_plots(logs_dir, base_dir, scenario)

    print(f"\n{'='*62}")
    print("  Done. Results in output/")
    print("=" * 62)


if __name__ == "__main__":
    main()
