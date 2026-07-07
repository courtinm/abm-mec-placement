"""
compare_strategies_urban_medium.py
------------------------------------
Compares the 5 CR placement strategies (No-CR, Random, Static,
Greedy/oracle, RL Agent) on the urban_medium scenario: mean global
satisfaction rate +/- 95% CI over N_SEEDS seeds, 300 steps/run with the
first 50 (warm-up) excluded.

Reuses the RN/CR Q-tables already trained by generate_results.py
(resultats/resultats-urban_medium/models/). Does NOT retrain anything --
if those Q-table files are missing, the script stops with an error message
telling you to train first.

Each (strategy, seed) eval run is cached: if its satisfaction_summary.csv
already exists under OUTPUT_ROOT, it is reused instead of re-simulated.
This means bumping N_SEEDS from 5 to 20 later only simulates the 15 new
seeds, not all 20 again.

Usage
-----
    python compare_strategies_urban_medium.py

Outputs (in baseline_comparison/urban_medium/)
------------------------------------------------
    eval/{strategy}/s{seed}/satisfaction_summary.csv   (per-run logs, cached)
    strategy_comparison_n{N_SEEDS}.csv                 (aggregated table)
    strategy_comparison_n{N_SEEDS}.png                 (bar chart)
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from main import build_simulation
from agents.cr_placement_agent import CRPlacementAgent
from agents.placement_strategies import make_strategy

# ── Configuration -- change N_SEEDS here and re-run ─────────────────────────
N_SEEDS      = 20
SEEDS        = list(range(N_SEEDS))

SCENARIO     = "urban_medium"
EVAL_STEPS   = 300
WARMUP       = 50
DPI          = 300

MODELS_DIR   = os.path.join("resultats", "resultats-urban_medium", "models")
OUTPUT_ROOT  = os.path.join("baseline_comparison", "urban_medium")
EVAL_DIR     = os.path.join(OUTPUT_ROOT, "eval")

BASELINE_STRATEGIES = ["no_cr", "random", "static", "exhaustive_greedy"]
ALL_STRATEGY_KEYS   = BASELINE_STRATEGIES + ["trained"]

STRATEGY_LABELS = {
    "no_cr":             "No-CR",
    "random":            "Random",
    "static":            "Static",
    "exhaustive_greedy": "Greedy (oracle)",
    "trained":           "RL Agent",
}
STRATEGY_COLORS = {
    "no_cr":             "#95a5a6",
    "random":            "#e74c3c",
    "static":            "#f39c12",
    "exhaustive_greedy": "#27ae60",
    "trained":           "#2980b9",
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 100})


# ── Utilities ────────────────────────────────────────────────────────────────

def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def _ci95(values):
    n = len(values)
    return 0.0 if n < 2 else 1.96 * float(np.std(values, ddof=1)) / np.sqrt(n)


def _rn_model_path(rn_id):
    return os.path.join(MODELS_DIR, f"rn_{rn_id}.pkl")


def _check_models_exist(config):
    missing = []
    for i in range(len(config["relay_nodes"])):
        p = _rn_model_path(i + 1)
        if not os.path.exists(p):
            missing.append(p)
    cr_path = os.path.join(MODELS_DIR, "cr_qtable.pkl")
    if not os.path.exists(cr_path):
        missing.append(cr_path)
    if missing:
        raise SystemExit(
            "Missing trained Q-table(s), nothing was simulated:\n  "
            + "\n  ".join(missing)
            + "\nTrain them first, e.g.: python generate_results.py --scenarios urban_medium"
        )


# ── Simulation runners (mirrors generate_results.py's eval logic) ──────────

def _freeze_rn(sim):
    for rn in sim.relay_nodes:
        path = _rn_model_path(rn.id)
        if os.path.exists(path):
            rn.agent.load_qtable(path)
            rn.agent.frozen = True
            rn.agent.epsilon = 0.0


def run_eval_trained(seed, config, out_dir):
    _seed_all(seed)
    sim = build_simulation(config)
    sim.dynamic_rn = False
    _freeze_rn(sim)

    for bs in sim.base_stations:
        bs.has_compute_resource = False
        bs.compute_resource = None

    cr_cfg = config.get("cr_placement", {})
    agent = CRPlacementAgent(sim.base_stations, cr_cfg.get("k", 2), cr_cfg.get("cr_capacity_mbps", 100.0))
    agent._users = sim.users
    agent._relay_nodes = sim.relay_nodes
    cr_path = os.path.join(MODELS_DIR, "cr_qtable.pkl")
    agent.load_qtable(cr_path)
    agent.frozen = True
    agent.epsilon = 0.0
    sim.cr_agent = agent

    for _ in range(EVAL_STEPS):
        sim.simulate_step()
    sim.finalize(output_dir=out_dir)


def run_eval_baseline(strategy_name, seed, config, out_dir):
    _seed_all(seed)
    sim = build_simulation(config)
    sim.dynamic_rn = False
    _freeze_rn(sim)

    for bs in sim.base_stations:
        bs.has_compute_resource = False
        bs.compute_resource = None

    cr_cfg = config.get("cr_placement", {})
    strategy = make_strategy(
        strategy_name, sim.base_stations,
        k=cr_cfg.get("k", 2), cr_capacity_mbps=cr_cfg.get("cr_capacity_mbps", 100.0),
    )
    if hasattr(strategy, "_users"):
        strategy._users = sim.users
    sim.cr_agent = strategy

    for _ in range(EVAL_STEPS):
        sim.simulate_step()
    sim.finalize(output_dir=out_dir)


def ensure_run(strategy_key, seed, config):
    """Run eval for (strategy_key, seed) unless its output is already cached."""
    out_dir = os.path.join(EVAL_DIR, strategy_key, f"s{seed}")
    csv_path = os.path.join(out_dir, "satisfaction_summary.csv")
    if os.path.exists(csv_path):
        return "cached"

    os.makedirs(out_dir, exist_ok=True)
    if strategy_key == "trained":
        run_eval_trained(seed, config, out_dir)
    else:
        run_eval_baseline(strategy_key, seed, config, out_dir)
    return "simulated"


# ── Aggregation ──────────────────────────────────────────────────────────────

def collect_mean_satisfaction(strategy_key):
    """Per-seed post-warmup mean Rate_Global for a strategy."""
    means = []
    for seed in SEEDS:
        path = os.path.join(EVAL_DIR, strategy_key, f"s{seed}", "satisfaction_summary.csv")
        df = pd.read_csv(path)
        df = df[df["Step"] > WARMUP]
        means.append(float(df["Rate_Global"].mean()))
    return means


def collect_timeseries(strategy_key):
    """Per-step, cross-seed mean and 95% CI of Rate_Global (post-warmup)."""
    per_seed = []
    for seed in SEEDS:
        path = os.path.join(EVAL_DIR, strategy_key, f"s{seed}", "satisfaction_summary.csv")
        df = pd.read_csv(path)
        df = df[df["Step"] > WARMUP][["Step", "Rate_Global"]].set_index("Step")
        per_seed.append(df)
    combined = pd.concat(per_seed, axis=1)
    combined.columns = range(len(combined.columns))
    mean = combined.mean(axis=1)
    ci95 = combined.apply(lambda row: _ci95(row.dropna().tolist()), axis=1)
    return mean, ci95


# ── Plot ─────────────────────────────────────────────────────────────────────

def make_bar_chart(rows, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(rows))
    means = [r["mean"] for r in rows]
    cis = [r["ci95"] for r in rows]
    colors = [STRATEGY_COLORS[r["strategy"]] for r in rows]
    labels = [STRATEGY_LABELS[r["strategy"]] for r in rows]

    bars = ax.bar(x, means, yerr=cis, capsize=7, color=colors, alpha=0.87,
                  edgecolor="white", linewidth=1.2,
                  error_kw={"elinewidth": 2, "ecolor": "#333333", "capthick": 2})
    for bar, m, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, m + ci + 0.015,
                f"{m:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(1.0, color="grey", linestyle="--", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Mean satisfaction rate (post-warmup)")
    ax.set_title(
        f"Strategy comparison -- urban_medium\n"
        f"Global satisfaction +/- 95% CI  (n = {N_SEEDS} seeds)",
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def make_timeseries_chart(timeseries, out_path, ma_window=15):
    """Line per strategy (rolling mean) with a shaded 95% CI ribbon -- makes
    overlapping strategies visually obvious as overlapping bands."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for strategy_key in ALL_STRATEGY_KEYS:
        mean, ci95 = timeseries[strategy_key]
        steps = mean.index.to_numpy()
        mean_s = mean.rolling(ma_window, center=True, min_periods=1).mean().to_numpy()
        ci_s = ci95.rolling(ma_window, center=True, min_periods=1).mean().to_numpy()
        color = STRATEGY_COLORS[strategy_key]
        label = STRATEGY_LABELS[strategy_key]

        ax.plot(steps, mean_s, color=color, lw=2.2,
                label=label, zorder=3 if strategy_key == "trained" else 2)
        ax.fill_between(steps, mean_s - ci_s, mean_s + ci_s, color=color, alpha=0.15, zorder=1)

    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Evaluation step (post-warmup)")
    ax.set_ylabel("Satisfaction rate")
    ax.set_title(
        f"Strategy comparison over time -- urban_medium\n"
        f"Rolling mean (w={ma_window}) +/- 95% CI band  (n = {N_SEEDS} seeds)",
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def make_forest_plot(rows, out_path):
    """Horizontal dot-and-whisker plot, sorted by mean -- overlapping CI
    whiskers between neighbouring strategies are immediately visible."""
    ranked = sorted(rows, key=lambda r: r["mean"], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(ranked))[::-1]  # best strategy at the top

    for yi, r in zip(y, ranked):
        color = STRATEGY_COLORS[r["strategy"]]
        ax.errorbar(r["mean"], yi, xerr=r["ci95"], fmt="o", color=color,
                    markersize=9, capsize=6, elinewidth=2.5, capthick=2.5, zorder=3)

    # Shade the oracle's CI band across the whole plot as a reference ceiling
    oracle = next(r for r in rows if r["strategy"] == "exhaustive_greedy")
    ax.axvspan(oracle["mean"] - oracle["ci95"], oracle["mean"] + oracle["ci95"],
               color=STRATEGY_COLORS["exhaustive_greedy"], alpha=0.08, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels([STRATEGY_LABELS[r["strategy"]] for r in ranked], fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Mean satisfaction rate (post-warmup)")
    ax.set_title(
        f"Strategy ranking -- urban_medium\n"
        f"Mean +/- 95% CI, sorted  (n = {N_SEEDS} seeds)",
        fontweight="bold",
    )
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── CI-overlap summary ───────────────────────────────────────────────────────

def print_overlap_summary(rows):
    ranked = sorted(rows, key=lambda r: r["mean"], reverse=True)
    print("\nRanking (best to worst) and pairwise CI overlap between neighbours:")
    for r in ranked:
        lo, hi = r["mean"] - r["ci95"], r["mean"] + r["ci95"]
        print(f"  {STRATEGY_LABELS[r['strategy']]:<18} mean={r['mean']:.3f}  "
              f"95% CI [{lo:.3f}, {hi:.3f}]")

    print()
    for a, b in zip(ranked, ranked[1:]):
        a_lo, a_hi = a["mean"] - a["ci95"], a["mean"] + a["ci95"]
        b_lo, b_hi = b["mean"] - b["ci95"], b["mean"] + b["ci95"]
        overlap = (a_lo <= b_hi) and (b_lo <= a_hi)
        verdict = "CI overlap -> NOT statistically distinguishable" if overlap \
            else "no CI overlap -> distinguishable"
        print(f"  {STRATEGY_LABELS[a['strategy']]:<18} vs {STRATEGY_LABELS[b['strategy']]:<18} : {verdict}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import importlib
    config = importlib.import_module(f"configs.{SCENARIO}").CONFIG

    _check_models_exist(config)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print(f"Scenario: {SCENARIO}  |  N_SEEDS={N_SEEDS}  |  steps={EVAL_STEPS} (warmup={WARMUP})")
    for strategy_key in ALL_STRATEGY_KEYS:
        print(f"\n[{STRATEGY_LABELS[strategy_key]}]")
        for seed in SEEDS:
            status = ensure_run(strategy_key, seed, config)
            print(f"  seed {seed}: {status}")

    rows = []
    for strategy_key in ALL_STRATEGY_KEYS:
        seed_means = collect_mean_satisfaction(strategy_key)
        rows.append({
            "strategy": strategy_key,
            "mean": float(np.mean(seed_means)),
            "std": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
            "ci95": _ci95(seed_means),
            "n": len(seed_means),
        })

    summary_csv = os.path.join(OUTPUT_ROOT, f"strategy_comparison_n{N_SEEDS}.csv")
    out_df = pd.DataFrame([
        {
            "strategy": STRATEGY_LABELS[r["strategy"]],
            "mean": round(r["mean"], 4),
            "std": round(r["std"], 4),
            "ci95": round(r["ci95"], 4),
            "n": r["n"],
        }
        for r in rows
    ])
    out_df.to_csv(summary_csv, index=False)
    print(f"\nSaved: {summary_csv}")

    fig_path = os.path.join(OUTPUT_ROOT, f"strategy_comparison_n{N_SEEDS}.png")
    make_bar_chart(rows, fig_path)
    print(f"Saved: {fig_path}")

    timeseries = {key: collect_timeseries(key) for key in ALL_STRATEGY_KEYS}
    ts_path = os.path.join(OUTPUT_ROOT, f"strategy_timeseries_n{N_SEEDS}.png")
    make_timeseries_chart(timeseries, ts_path)
    print(f"Saved: {ts_path}")

    forest_path = os.path.join(OUTPUT_ROOT, f"strategy_forest_n{N_SEEDS}.png")
    make_forest_plot(rows, forest_path)
    print(f"Saved: {forest_path}")

    print_overlap_summary(rows)


if __name__ == "__main__":
    main()
