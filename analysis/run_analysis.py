"""
Run all analysis from experiment CSV logs.

Usage:
    python analysis/run_analysis.py                        # default: logs/ -> plots/
    python analysis/run_analysis.py --logs-dir logs/ --output-dir plots/ --warmup 50

(always run from the project root, abm_communication_networks/)
"""

import argparse
import csv
import os
import statistics
import sys

# Allow `python analysis/run_analysis.py` to find the project-root packages
# (plots/, stats_utils) regardless of the current directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stats_utils import ci95 as _ci95


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_col(path, col):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return [float(r[col]) for r in csv.DictReader(f)]


def _mean_ci(values):
    if not values:
        return None, None
    m = statistics.mean(values)
    return m, _ci95(values)


def _section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print('='*62)


# ── 1. Training curves ────────────────────────────────────────────────────────

def print_training_stats(logs_dir):
    _section("TRAINING - learning curves")

    rn_path = os.path.join(logs_dir, "train_rn", "rn_reward.csv")
    rn = _load_col(rn_path, "TotalReward")
    if rn:
        n = len(rn)
        first, last = rn[:n//5], rn[-n//5:]
        trend = "IMPROVING" if statistics.mean(last) > statistics.mean(first) else "flat"
        print(f"\nRN agent  ({n} steps)")
        print(f"  First 20% : mean = {statistics.mean(first):.3f}")
        print(f"  Last  20% : mean = {statistics.mean(last):.3f}  -> {trend}")
    else:
        print("\nRN reward  : not found (run --mode train_rn first)")

    cr_path = os.path.join(logs_dir, "train_cr", "cr_reward.csv")
    dq_path = os.path.join(logs_dir, "train_cr", "cr_delta_q.csv")
    cr = _load_col(cr_path, "SatisfactionRate")
    dq = _load_col(dq_path, "DeltaQ")
    if cr:
        n = len(cr)
        first, last = cr[:n//5], cr[-n//5:]
        trend = "IMPROVING" if statistics.mean(last) > statistics.mean(first) else "flat"
        print(f"\nCR agent  ({n} steps logged, warmup excluded)")
        print(f"  First 20% : mean = {statistics.mean(first):.3f}")
        print(f"  Last  20% : mean = {statistics.mean(last):.3f}  -> {trend}")
        if dq:
            ratio = statistics.mean(dq[-100:]) / statistics.mean(dq[:100])
            print(f"  DeltaQ convergence : {ratio:.3f}x of initial"
                  f"  ({'converging' if ratio < 0.5 else 'still learning'})")
    else:
        print("\nCR reward  : not found (run --mode train_cr first)")


# ── 2. Eval comparison ────────────────────────────────────────────────────────

STRATEGIES = ["no_cr", "random", "static", "exhaustive_greedy", "trained"]
SCENARIOS  = ["urban_light", "urban_medium", "urban_dense"]


def _collect_seeds(strategy_dir, warmup):
    """Walk strategy_dir, collect per-seed mean satisfaction."""
    means = []
    for root, _, files in os.walk(strategy_dir):
        if "satisfaction_summary.csv" not in files:
            continue
        rows = [r for r in csv.DictReader(open(os.path.join(root, "satisfaction_summary.csv")))
                if int(r["Step"]) > warmup]
        if rows:
            means.append(statistics.mean(float(r["Rate_Global"]) for r in rows))
    return means


def _collect_per_app(strategy_dir, warmup):
    """Return {app: [seed_means]} for AR_VR, Streaming, BestEffort."""
    cols = {"Rate_AR_VR": [], "Rate_Streaming": [], "Rate_BestEffort": []}
    for root, _, files in os.walk(strategy_dir):
        if "satisfaction_summary.csv" not in files:
            continue
        rows = [r for r in csv.DictReader(open(os.path.join(root, "satisfaction_summary.csv")))
                if int(r["Step"]) > warmup]
        if not rows:
            continue
        for col in cols:
            cols[col].append(statistics.mean(float(r[col]) for r in rows))
    return cols


def print_eval_table(logs_dir, warmup):
    eval_dir = os.path.join(logs_dir, "eval")

    _section("EVAL - global satisfaction rate (mean +/- 95% CI over seeds)")
    print(f"\n{'Strategy':<22}", end="")
    for sc in SCENARIOS:
        print(f"{sc:>18}", end="")
    print()
    print("-" * (22 + 18 * len(SCENARIOS)))

    for strat in STRATEGIES:
        print(f"{strat:<22}", end="")
        for sc in SCENARIOS:
            if strat == "trained":
                d = os.path.join(eval_dir, "trained", sc)
            else:
                d = os.path.join(eval_dir, strat, sc)

            if not os.path.isdir(d):
                print(f"{'---':>18}", end="")
                continue

            seeds = _collect_seeds(d, warmup)
            if not seeds:
                print(f"{'no data':>18}", end="")
                continue

            m, ci = _mean_ci(seeds)
            n = len(seeds)
            cell = f"{m:.3f} +/- {ci:.3f} (n={n})"
            print(f"{cell:>18}", end="")
        print()

    _section("EVAL - per-app satisfaction (trained agent only)")
    app_labels = {"Rate_AR_VR": "AR/VR", "Rate_Streaming": "Streaming",
                  "Rate_BestEffort": "Best-effort"}
    print(f"\n{'App type':<15}", end="")
    for sc in SCENARIOS:
        print(f"{sc:>20}", end="")
    print()
    print("-" * (15 + 20 * len(SCENARIOS)))

    for col, label in app_labels.items():
        print(f"{label:<15}", end="")
        for sc in SCENARIOS:
            d = os.path.join(eval_dir, "trained", sc)
            if not os.path.isdir(d):
                print(f"{'---':>20}", end="")
                continue
            data = _collect_per_app(d, warmup)
            vals = data.get(col, [])
            if not vals:
                print(f"{'no data':>20}", end="")
                continue
            m, ci = _mean_ci(vals)
            cell = f"{m:.3f} +/- {ci:.3f} (n={len(vals)})"
            print(f"{cell:>20}", end="")
        print()


# ── 3. Figures ────────────────────────────────────────────────────────────────

def generate_figures(logs_dir, output_dir):
    _section("FIGURES - generating plots")
    from plots.plot_results import (
        fig1_training_curves,
        fig2_delta_q,
        fig3_strategy_comparison,
        fig4_per_app_satisfaction,
    )
    os.makedirs(output_dir, exist_ok=True)
    fig1_training_curves(logs_dir, output_dir)
    fig2_delta_q(logs_dir, output_dir)
    fig3_strategy_comparison(logs_dir, output_dir, warmup=50)
    fig4_per_app_satisfaction(logs_dir, output_dir, warmup=50)
    print(f"\nAll figures saved to '{output_dir}/'")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Print training stats, eval comparison table, and generate figures."
    )
    parser.add_argument("--logs-dir",   default="logs/")
    parser.add_argument("--output-dir", default="plots/")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation (print stats only)")
    args = parser.parse_args()

    print_training_stats(args.logs_dir)
    print_eval_table(args.logs_dir, args.warmup)

    if not args.no_figures:
        generate_figures(args.logs_dir, args.output_dir)


if __name__ == "__main__":
    main()
