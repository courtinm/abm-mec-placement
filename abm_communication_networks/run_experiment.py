import argparse
import csv
import importlib
import os
import random

from main import build_simulation
from agents.placement_strategies import STRATEGY_NAMES, make_strategy

SCENARIOS = ("urban", "suburban", "rural", "default")


def load_config(scenario):
    module = importlib.import_module(f"configs.{scenario}")
    return module.CONFIG


def _rn_path(base_path, rn_id):
    """Derive per-RN path: models/rn.pkl -> models/rn_1.pkl"""
    stem, ext = os.path.splitext(base_path)
    return f"{stem}_{rn_id}{ext}"


def _attach_cr_agent(sim, config, qtable_path=None, freeze=False):
    """Create a CRPlacementAgent, attach it to the simulator, and return it."""
    from agents.cr_placement_agent import CRPlacementAgent

    cr_cfg = config.get("cr_placement", {})
    k = cr_cfg.get("k", 2)
    cr_capacity = cr_cfg.get("cr_capacity_mbps", 100.0)

    for bs in sim.base_stations:
        bs.has_compute_resource = False
        bs.compute_resource = None

    agent = CRPlacementAgent(sim.base_stations, k, cr_capacity)
    agent._users = sim.users
    agent._relay_nodes = sim.relay_nodes

    if qtable_path is not None:
        agent.load_qtable(qtable_path)

    if freeze:
        agent.frozen = True
        agent.epsilon = 0.0

    sim.cr_agent = agent
    return agent


def _attach_strategy(sim, config, name):
    """Create a baseline strategy, attach it, and return it."""
    cr_cfg = config.get("cr_placement", {})
    k = cr_cfg.get("k", 2)
    cr_capacity = cr_cfg.get("cr_capacity_mbps", 100.0)

    for bs in sim.base_stations:
        bs.has_compute_resource = False
        bs.compute_resource = None

    strategy = make_strategy(name, sim.base_stations, k=k, cr_capacity_mbps=cr_capacity)

    # GreedyOptimalStrategy needs a live reference to the users list
    if hasattr(strategy, "_users"):
        strategy._users = sim.users

    sim.cr_agent = strategy
    return strategy


def main():
    parser = argparse.ArgumentParser(description="Run headless simulation (no pygame)")
    parser.add_argument("--scenario", choices=SCENARIOS, default="urban")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("train_rn", "train_cr", "eval"), default="eval",
        help="train_rn: learn RN placement; train_cr: freeze RN, learn CR; "
             "eval: measure (use with --strategy OR --cr-qtable)",
    )
    parser.add_argument(
        "--strategy", choices=STRATEGY_NAMES, default=None, metavar="STRATEGY",
        help=f"Baseline CR placement strategy for --mode eval. "
             f"Choices: {STRATEGY_NAMES}. "
             "When set, --cr-qtable is not required.",
    )
    parser.add_argument(
        "--rn-qtable", default=None, metavar="PATH",
        help="Base path for per-RN Q-table files. "
             "Required for train_cr/eval without --strategy. "
             "Optional (but recommended) with --strategy.",
    )
    parser.add_argument(
        "--cr-qtable", default=None, metavar="PATH",
        help="Path for CR placement Q-table. "
             "Required for eval without --strategy. "
             "Saved on train_cr.",
    )
    parser.add_argument("--warmup", type=int, default=50,
                        help="Steps excluded from cr_reward.csv logging (default: 50)")
    args = parser.parse_args()

    # ── Argument validation ───────────────────────────────────────────
    if args.strategy is not None and args.mode != "eval":
        parser.error("--strategy is only valid with --mode eval")

    if args.strategy is None:
        # RL path: Q-tables required
        if args.mode in ("train_cr", "eval") and args.rn_qtable is None:
            parser.error(f"--rn-qtable is required for mode '{args.mode}' "
                         "(or use --strategy for a baseline run)")
        if args.mode == "eval" and args.cr_qtable is None:
            parser.error("--cr-qtable is required for mode 'eval' "
                         "(or use --strategy for a baseline run)")

    # ── Setup ─────────────────────────────────────────────────────────
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.strategy is not None:
        # logs/eval/<strategy>/<scenario>/s<seed>/  — expected by plot_results.py
        output_dir = os.path.join("logs", "eval", args.strategy,
                                  args.scenario, f"s{args.seed}")
    elif args.mode == "eval":
        # Trained RL agent: logs/eval/trained/<scenario>/s<seed>/
        output_dir = os.path.join("logs", "eval", "trained",
                                  args.scenario, f"s{args.seed}")
    elif args.mode in ("train_rn", "train_cr"):
        # logs/train_rn/  or  logs/train_cr/
        output_dir = os.path.join("logs", args.mode)
    else:
        output_dir = os.path.join("logs", args.scenario)
    os.makedirs(output_dir, exist_ok=True)

    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    config = load_config(args.scenario)
    sim = build_simulation(config)
    sim.dynamic_rn = False

    # ── Agent / strategy setup ────────────────────────────────────────
    if args.strategy is not None:
        # Baseline eval: no Q-tables required
        if args.rn_qtable is not None:
            # Optionally freeze RN with a pre-trained Q-table
            for rn in sim.relay_nodes:
                path = _rn_path(args.rn_qtable, rn.id)
                if os.path.exists(path):
                    rn.agent.load_qtable(path)
                    rn.agent.frozen = True
                    rn.agent.epsilon = 0.0
            print(f"[eval/{args.strategy}] RN Q-tables loaded and frozen.")

        _attach_strategy(sim, config, args.strategy)
        print(f"[eval] Strategy: {args.strategy}  "
              f"(K={config.get('cr_placement', {}).get('k', 2)} CRs / "
              f"{len(sim.base_stations)} BS)")

    elif args.mode in ("train_cr", "eval"):
        for rn in sim.relay_nodes:
            rn.agent.load_qtable(_rn_path(args.rn_qtable, rn.id))
            rn.agent.frozen = True
            rn.agent.epsilon = 0.0
        print(f"[{args.mode}] RN Q-tables loaded and frozen.")

        freeze_cr = (args.mode == "eval")
        _attach_cr_agent(sim, config,
                         qtable_path=args.cr_qtable if freeze_cr else None,
                         freeze=freeze_cr)
        mode_label = "frozen" if freeze_cr else "learning"
        print(f"[{args.mode}] CR agent ready ({mode_label}).")

    # ── Simulation loop ───────────────────────────────────────────────
    rn_reward_rows = []
    cr_reward_rows = []

    for _ in range(args.steps):
        sim.simulate_step()
        step = sim.timestep

        if args.mode == "train_rn":
            total_rn_reward = sum(rn.last_reward for rn in sim.relay_nodes)
            rn_reward_rows.append((step, total_rn_reward))

        if (args.mode in ("train_cr", "eval") or args.strategy is not None) \
                and step > args.warmup \
                and sim.cr_agent is not None \
                and sim.cr_agent.last_reward is not None:
            cr_reward_rows.append((step, round(sim.cr_agent.last_reward, 4)))

    sim.finalize(output_dir=output_dir)

    # ── Save artifacts ────────────────────────────────────────────────
    if args.mode == "train_rn":
        if args.rn_qtable is not None:
            for rn in sim.relay_nodes:
                rn.agent.save_qtable(_rn_path(args.rn_qtable, rn.id))
            print(f"RN Q-tables saved (base: '{args.rn_qtable}').")

        reward_path = os.path.join(output_dir, "rn_reward.csv")
        with open(reward_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "TotalReward"])
            writer.writerows(rn_reward_rows)
        print(f"RN reward log -> '{reward_path}'.")

    if args.mode == "train_cr":
        if args.cr_qtable is not None:
            sim.cr_agent.save_qtable(args.cr_qtable)
            print(f"CR Q-table saved to '{args.cr_qtable}'.")

        reward_path = os.path.join(output_dir, "cr_reward.csv")
        with open(reward_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "SatisfactionRate"])
            writer.writerows(cr_reward_rows)
        print(f"CR reward log -> '{reward_path}'.")

    print(f"Done. {args.steps} steps | scenario={args.scenario} | "
          f"mode={args.mode} | strategy={args.strategy or 'RL'} | seed={args.seed}. "
          f"Logs -> '{output_dir}'.")


if __name__ == "__main__":
    main()
