"""
compare_migration_cost.py
------------------------------
Before/after/alternative comparison of the CR migration-cost models
(dissertation Ch.7): no cost model (None, current default) vs. "hard_cutover"
vs. "make_before_break" — see simulation/simulator.py's migration_cost_mode
and run_experiment.py's --migration-mode.

For a given scenario and seed, runs --mode eval three times (one per
migration mode) for each of the five placement strategies (No-CR, Random,
Static, Greedy/oracle, RL Agent), reusing the RN/CR Q-tables already trained
by generate_results.py (resultats/resultats-{scenario}/models/). Does NOT
retrain anything and does NOT touch train_rn/train_cr -- migration_cost_mode
is only ever set for the eval runs here.

RN Q-tables are always loaded and frozen (--rn-qtable equivalent): this
matters because an *unfrozen* RN agent's reward depends on user.is_satisfied
(agents/relay_node.py), which the "hard_cutover" mode can force to False --
with an unfrozen RN that would feed back into RN Q-learning and make the
comparison confounded by diverging RN trajectories. With RN frozen (the
standard eval setup, matching generate_results.py), the three migration
modes are verified to produce byte-identical simulation trajectories
(hop_counts.csv) and differ only in the satisfaction_users_log /
satisfaction_summary_log overlay -- see the control run in the session that
introduced this script.

Each (scenario, strategy, mode, seed) eval run is cached under OUTPUT_ROOT:
if its satisfaction_summary.csv already exists, it is reused instead of
re-simulated.

Usage
-----
    python compare_migration_cost.py                                   # urban_medium, seed 0, all 5 strategies
    python compare_migration_cost.py --scenarios urban_medium urban_dense --seeds 0 1 2
    python compare_migration_cost.py --strategies random trained        # subset of strategies

Outputs (in migration_cost_comparison/)
------------------------------------------
    eval/{scenario}/{mode}/{strategy}/s{seed}/satisfaction_summary.csv  (per-run logs, cached)
    migration_cost_comparison.csv   (long-format table: scenario, mode, strategy, app, mean, ci95, n)
    Printed comparison table (global + per-app satisfaction, 3 modes side by side, per strategy)
"""

import argparse
import importlib
import os

import numpy as np
import pandas as pd

from main import build_simulation
from agents.cr_placement_agent import CRPlacementAgent
from agents.placement_strategies import make_strategy

SCENARIOS_DEFAULT = ["urban_medium"]
SEEDS_DEFAULT      = [0]
STRATEGIES_DEFAULT = ["no_cr", "random", "static", "exhaustive_greedy", "trained"]
MODES = [None, "hard_cutover", "make_before_break"]
MODE_LABELS = {None: "None (current)", "hard_cutover": "hard_cutover", "make_before_break": "make_before_break"}

APP_COLS   = ["Rate_Global", "Rate_AR_VR", "Rate_Streaming", "Rate_BestEffort"]
APP_LABELS = {
    "Rate_Global":     "Global",
    "Rate_AR_VR":      "AR/VR",
    "Rate_Streaming":  "Streaming",
    "Rate_BestEffort": "Best-effort",
}
STRATEGY_LABELS = {
    "no_cr":             "No-CR",
    "random":            "Random",
    "static":            "Static",
    "exhaustive_greedy": "Greedy (oracle)",
    "trained":           "RL Agent",
}

WARMUP     = 50
EVAL_STEPS = 300

OUTPUT_ROOT = "migration_cost_comparison"
EVAL_DIR    = os.path.join(OUTPUT_ROOT, "eval")


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
    import random
    random.seed(seed)
    np.random.seed(seed)


def _ci95(values):
    n = len(values)
    return 0.0 if n < 2 else 1.96 * float(np.std(values, ddof=1)) / np.sqrt(n)


def _freeze_rn(sim, models_dir):
    for rn in sim.relay_nodes:
        path = _rn_model_path(models_dir, rn.id)
        if os.path.exists(path):
            rn.agent.load_qtable(path)
            rn.agent.frozen = True
            rn.agent.epsilon = 0.0


def run_eval(scenario, strategy_key, mode, seed, models_dir, out_dir):
    _seed_all(seed)
    config = importlib.import_module(f"configs.{scenario}").CONFIG
    sim = build_simulation(config)
    sim.dynamic_rn = False
    _freeze_rn(sim, models_dir)  # frozen RN -- see module docstring for why this matters here

    for bs in sim.base_stations:
        bs.has_compute_resource = False
        bs.compute_resource = None

    cr_cfg = config.get("cr_placement", {})
    if strategy_key == "trained":
        agent = CRPlacementAgent(sim.base_stations, cr_cfg.get("k", 2), cr_cfg.get("cr_capacity_mbps", 100.0))
        agent._users = sim.users
        agent._relay_nodes = sim.relay_nodes
        agent.load_qtable(os.path.join(models_dir, "cr_qtable.pkl"))
        agent.frozen = True
        agent.epsilon = 0.0
        sim.cr_agent = agent
    else:
        strategy = make_strategy(strategy_key, sim.base_stations,
                                 k=cr_cfg.get("k", 2), cr_capacity_mbps=cr_cfg.get("cr_capacity_mbps", 100.0))
        if hasattr(strategy, "_users"):
            strategy._users = sim.users
        sim.cr_agent = strategy

    sim.migration_cost_mode = mode  # None | "hard_cutover" | "make_before_break"

    for _ in range(EVAL_STEPS):
        sim.simulate_step()
    sim.finalize(output_dir=out_dir)


def ensure_run(scenario, strategy_key, mode, seed, models_dir):
    mode_key = mode or "none"
    out_dir = os.path.join(EVAL_DIR, scenario, mode_key, strategy_key, f"s{seed}")
    csv_path = os.path.join(out_dir, "satisfaction_summary.csv")
    if os.path.exists(csv_path):
        return "cached"
    os.makedirs(out_dir, exist_ok=True)
    run_eval(scenario, strategy_key, mode, seed, models_dir, out_dir)
    return "simulated"


def _per_app_means(scenario, strategy_key, mode, seeds):
    mode_key = mode or "none"
    per_app = {c: [] for c in APP_COLS}
    for seed in seeds:
        path = os.path.join(EVAL_DIR, scenario, mode_key, strategy_key, f"s{seed}", "satisfaction_summary.csv")
        df = pd.read_csv(path)
        df = df[df["Step"] > WARMUP]
        for c in APP_COLS:
            per_app[c].append(float(df[c].mean()))
    return per_app


def print_comparison_table(rows_df, scenario, strategies):
    print(f"\n{'=' * 90}\n  {scenario} -- migration cost comparison (n={rows_df['n'].iloc[0]} seed(s))\n{'=' * 90}")
    for strategy_key in strategies:
        print(f"\n  [{STRATEGY_LABELS[strategy_key]}]")
        header = f"    {'App':<12}" + "".join(f"{MODE_LABELS[m]:>20}" for m in MODES)
        print(header)
        sub = rows_df[(rows_df["scenario"] == scenario) & (rows_df["strategy"] == strategy_key)]
        for app_col in APP_COLS:
            line = f"    {APP_LABELS[app_col]:<12}"
            for mode in MODES:
                row = sub[(sub["app"] == APP_LABELS[app_col]) & (sub["mode"] == (mode or "none"))]
                if row.empty:
                    line += f"{'--':>20}"
                else:
                    r = row.iloc[0]
                    line += f"{r['mean']:.3f} +/- {r['ci95']:.3f}".rjust(20)
            print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS_DEFAULT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES_DEFAULT, choices=STRATEGIES_DEFAULT)
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_rows = []

    for scenario in args.scenarios:
        models_dir = os.path.join("resultats", f"resultats-{scenario}", "models")
        config = importlib.import_module(f"configs.{scenario}").CONFIG
        _check_models_exist(scenario, models_dir, config)

        print(f"\n{'=' * 62}\n  SCENARIO : {scenario.upper()}\n{'=' * 62}")
        for strategy_key in args.strategies:
            for mode in MODES:
                print(f"\n  [{STRATEGY_LABELS[strategy_key]} | {MODE_LABELS[mode]}]")
                for seed in args.seeds:
                    status = ensure_run(scenario, strategy_key, mode, seed, models_dir)
                    print(f"    seed {seed}: {status}")

        for strategy_key in args.strategies:
            for mode in MODES:
                per_app = _per_app_means(scenario, strategy_key, mode, args.seeds)
                for app_col in APP_COLS:
                    vals = per_app[app_col]
                    all_rows.append({
                        "scenario": scenario,
                        "strategy": strategy_key,
                        "mode": mode or "none",
                        "app": APP_LABELS[app_col],
                        "mean": round(float(np.mean(vals)), 4),
                        "ci95": round(_ci95(vals), 4),
                        "n": len(vals),
                    })

    rows_df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_ROOT, "migration_cost_comparison.csv")
    rows_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    for scenario in args.scenarios:
        print_comparison_table(rows_df, scenario, args.strategies)


if __name__ == "__main__":
    main()
