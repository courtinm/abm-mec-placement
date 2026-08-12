import argparse
import csv
import importlib
import os
import pickle
import random
import sys

# Allow `python experiments/run_experiment.py` to find the project-root
# packages (main.py, agents/, configs/) regardless of the current directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import build_simulation
from agents.placement_strategies import STRATEGY_NAMES, make_strategy

SCENARIOS = (
    "urban_light", "urban_medium", "urban_dense", "urban_controlled",
    "urban_large_N12", "urban_xl_lambda0", "urban_xl_lambda01",
    "urban", "default",
)

_ALIASES = {"urban": "urban_medium"}


def load_config(scenario):
    actual = _ALIASES.get(scenario, scenario)
    module = importlib.import_module(f"configs.{actual}")
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
    hp = cr_cfg.get("rl_hyperparams", {})

    for bs in sim.base_stations:
        bs.has_compute_resource = False
        bs.compute_resource = None

    agent = CRPlacementAgent(
        sim.base_stations, k, cr_capacity,
        learning_rate=hp.get("alpha_0", 0.1),
        epsilon=hp.get("epsilon_0", 0.5),
        epsilon_min=hp.get("epsilon_min", 0.05),
        epsilon_decay=hp.get("epsilon_decay", 0.995),
        alpha_min=hp.get("alpha_min", 0.01),
        alpha_decay=hp.get("alpha_decay", 0.998),
        reward_shaping_lambda=hp.get("reward_shaping_lambda", 0.0),
    )
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
    parser.add_argument("--scenario", choices=SCENARIOS, default="urban_medium")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("train_rn", "train_cr", "eval"), default="eval",
        help="train_rn: learn RN placement; train_cr: freeze RN, learn CR; "
             "eval: measure (use with --strategy OR --cr-qtable)",
    )
    parser.add_argument(
        "--strategy", choices=STRATEGY_NAMES + ["rl"], default=None, metavar="STRATEGY",
        help=f"Baseline CR placement strategy for --mode eval. "
             f"Choices: {STRATEGY_NAMES} (baseline) or 'rl' (trained agent). "
             "Use 'rl' with --cr-qtable to evaluate the trained RL agent.",
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
    parser.add_argument(
        "--migration-mode", choices=("hard_cutover", "make_before_break"), default=None,
        metavar="MODE",
        help="CR migration-cost model, --mode eval only (default: none, current "
             "behaviour). 'hard_cutover': users anchored to a BS that just lost "
             "its CR are forced unsatisfied for that step. 'make_before_break': "
             "does not affect satisfaction; logs migration_overhead_mbps instead.",
    )
    parser.add_argument(
        "--cr-admission-policy", choices=("fcfs", "priority"), default=None,
        metavar="POLICY",
        help="2x2 factorial (Ch.7 M0-M3), valid in ANY --mode since it changes "
             "the reward signal used during training. Default: scenario config's "
             "cr_admission_policy, itself 'fcfs' (M0) unless the config overrides it.",
    )
    parser.add_argument(
        "--radio-allocation", choices=("equal_share", "proportional"), default=None,
        metavar="ALLOCATION",
        help="2x2 factorial (Ch.7 M0-M3), valid in ANY --mode since it changes "
             "the reward signal used during training. Default: scenario config's "
             "radio_allocation, itself 'equal_share' (M0) unless the config overrides it.",
    )
    parser.add_argument(
        "--snapshot-every", type=int, default=0,
        help="If > 0, capture CR greedy-policy snapshots every N Q-updates during train_cr."
    )
    parser.add_argument(
        "--snapshot-min-state-visits", type=int, default=10,
        help="Minimum visits a state needs before it is included in a policy snapshot."
    )
    parser.add_argument(
        "--state-coverage-every", type=int, default=0,
        help="If > 0 and --mode is train_cr or eval, checkpoint CR state-space "
             "coverage every N steps to 'state_coverage_checkpoints.csv' in the "
             "output dir: states_in_qtable (len(q_table), an upper bound since "
             "next_state insertions count too) vs distinct_states_visited_cumulative "
             "(len(state_visit_counts), states actually chosen as the current state)."
    )
    parser.add_argument(
        "--track-eval-coverage", action="store_true",
        help="Eval mode + RL only. Snapshot the states present in the loaded "
             "(frozen) Q-table before the eval loop starts, then classify each "
             "post-warmup eval step's state as 'known' (was in that snapshot, "
             "i.e. seen during training) vs 'unknown' (never seen during "
             "training). Also tracks hits on --dominant-state if given. "
             "Writes 'eval_state_coverage.csv'."
    )
    parser.add_argument(
        "--dominant-state", default=None, metavar="A|B|C|...",
        help="Pipe-separated per-BS demand-level tuple (e.g. '2|2|2|2|2|2|2|2') "
             "to track hit-rate against, in addition to the known/unknown split. "
             "Only used with --track-eval-coverage."
    )
    args = parser.parse_args()

    # ── Argument validation ───────────────────────────────────────────
    _is_baseline = args.strategy is not None and args.strategy != "rl"
    if args.strategy is not None and args.strategy != "rl" and args.mode != "eval":
        parser.error("--strategy is only valid with --mode eval")
    if args.migration_mode is not None and args.mode != "eval":
        parser.error("--migration-mode is only valid with --mode eval "
                     "(migration cost is never modelled during train_rn/train_cr)")

    if not _is_baseline:
        # RL path: cr-qtable required for eval
        if args.mode == "eval" and args.rn_qtable is None and args.strategy != "rl":
            parser.error("--rn-qtable is required for mode 'eval' "
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
    sim = build_simulation(config)  # sets sim.cr_admission_policy / radio_allocation from config (M0 defaults)
    sim.dynamic_rn = False
    sim.migration_cost_mode = args.migration_mode  # None outside --mode eval (validated above)
    if args.cr_admission_policy is not None:
        sim.cr_admission_policy = args.cr_admission_policy
    if args.radio_allocation is not None:
        sim.radio_allocation = args.radio_allocation

    # ── Agent / strategy setup ────────────────────────────────────────
    if _is_baseline:
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
        if args.rn_qtable is not None:
            for rn in sim.relay_nodes:
                path = _rn_path(args.rn_qtable, rn.id)
                if os.path.exists(path):
                    rn.agent.load_qtable(path)
                    rn.agent.frozen = True
                    rn.agent.epsilon = 0.0
            print(f"[{args.mode}] RN Q-tables loaded and frozen.")
        else:
            print(f"[{args.mode}] No --rn-qtable provided - RNs use default policy.")

        freeze_cr = (args.mode == "eval")
        _attach_cr_agent(sim, config,
                         qtable_path=args.cr_qtable if freeze_cr else None,
                         freeze=freeze_cr)
        mode_label = "frozen" if freeze_cr else "learning"
        print(f"[{args.mode}] CR agent ready ({mode_label}).")

        if args.mode == "train_cr" and args.snapshot_every > 0:
            sim.cr_agent.policy_snapshot_history = []
            sim.cr_agent.state_visit_counts = {}

    # Snapshot of states known at load time (i.e. seen during training), taken
    # BEFORE the eval loop runs — select_action() inserts cold entries for any
    # novel state it encounters, so q_table.keys() would otherwise drift as
    # eval progresses and stop meaning "known from training".
    eval_known_states = None
    eval_dominant_state = None
    eval_total = 0
    eval_known_hits = 0
    eval_dominant_hits = 0
    if args.track_eval_coverage and args.mode == "eval" and hasattr(sim.cr_agent, "q_table"):
        eval_known_states = set(sim.cr_agent.q_table.keys())
        if args.dominant_state:
            eval_dominant_state = tuple(int(x) for x in args.dominant_state.split("|"))

    # ── Simulation loop ───────────────────────────────────────────────
    rn_reward_rows = []
    cr_reward_rows = []              # shaped reward (counterfactual + shaping)
    cr_reward_cf_rows = []           # counterfactual only
    cr_reward_shaping_rows = []      # lambda * r_latency only
    cr_reward_global_rows = []       # global satisfaction (for comparison)
    state_coverage_rows = []         # step, states_in_qtable, pct_of_3^n, distinct_visited_cumulative

    for _ in range(args.steps):
        sim.simulate_step()
        step = sim.timestep

        if (args.state_coverage_every > 0 and args.mode in ("train_cr", "eval")
                and step % args.state_coverage_every == 0
                and sim.cr_agent is not None
                and hasattr(sim.cr_agent, "q_table")):
            n_bs = getattr(sim.cr_agent, "n", None)
            max_states = 3 ** n_bs if n_bs else None
            states_in_qtable = len(sim.cr_agent.q_table)
            distinct_visited = len(getattr(sim.cr_agent, "state_visit_counts", {}))
            pct = round(100 * states_in_qtable / max_states, 6) if max_states else None
            state_coverage_rows.append((step, states_in_qtable, pct, distinct_visited))

        if args.mode == "train_rn":
            total_rn_reward = sum(rn.last_reward for rn in sim.relay_nodes)
            rn_reward_rows.append((step, total_rn_reward))

        if (args.mode in ("train_cr", "eval") or _is_baseline) \
                and step > args.warmup \
                and sim.cr_agent is not None \
                and sim.cr_agent.last_reward is not None:
            cr_reward_rows.append((step, round(sim.cr_agent.last_reward, 4)))
            if getattr(sim.cr_agent, "last_reward_counterfactual", None) is not None:
                cr_reward_cf_rows.append(
                    (step, round(sim.cr_agent.last_reward_counterfactual, 4))
                )
            if getattr(sim.cr_agent, "last_reward_shaping", None) is not None:
                cr_reward_shaping_rows.append(
                    (step, round(sim.cr_agent.last_reward_shaping, 4))
                )
            if getattr(sim.cr_agent, "last_reward_global", None) is not None:
                cr_reward_global_rows.append(
                    (step, round(sim.cr_agent.last_reward_global, 4))
                )

        if args.mode == "train_cr" and args.snapshot_every > 0 and sim.cr_agent is not None:
            if len(sim.cr_agent.delta_q_history) > 0 and len(sim.cr_agent.delta_q_history) % args.snapshot_every == 0:
                sim.cr_agent.capture_policy_snapshot(args.snapshot_min_state_visits)

        if eval_known_states is not None and step > args.warmup:
            s = sim.cr_agent.prev_state
            eval_total += 1
            if s in eval_known_states:
                eval_known_hits += 1
            if eval_dominant_state is not None and s == eval_dominant_state:
                eval_dominant_hits += 1

    sim.finalize(output_dir=output_dir)

    if state_coverage_rows:
        cov_path = os.path.join(output_dir, "state_coverage_checkpoints.csv")
        with open(cov_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "states_in_qtable", "pct_of_3^n", "distinct_states_visited_cumulative"])
            writer.writerows(state_coverage_rows)
        print(f"State coverage checkpoints -> '{cov_path}'.")

    if eval_known_states is not None:
        pct_known = round(100 * eval_known_hits / eval_total, 4) if eval_total else None
        pct_dominant = (round(100 * eval_dominant_hits / eval_total, 4)
                         if eval_total and eval_dominant_state is not None else None)
        eval_cov_path = os.path.join(output_dir, "eval_state_coverage.csv")
        with open(eval_cov_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["total_postwarmup_steps", "known_state_hits", "pct_known",
                              "dominant_state", "dominant_state_hits", "pct_dominant"])
            writer.writerow([
                eval_total, eval_known_hits, pct_known,
                args.dominant_state or "", eval_dominant_hits, pct_dominant,
            ])
        print(f"Eval state coverage -> '{eval_cov_path}'.")

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

        if getattr(sim.cr_agent, "state_visit_counts", None):
            visits_path = os.path.join(output_dir, "cr_state_visit_counts.csv")
            with open(visits_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["state", "visits"])
                for state, count in sim.cr_agent.state_visit_counts.items():
                    writer.writerow(["|".join(map(str, state)), count])
            print(f"CR state visit counts -> '{visits_path}'.")

        if args.snapshot_every > 0 and getattr(sim.cr_agent, "policy_snapshot_history", None):
            snapshot_dir = os.path.join(output_dir, "policy_snapshots")
            os.makedirs(snapshot_dir, exist_ok=True)
            for idx, snapshot in enumerate(sim.cr_agent.policy_snapshot_history, 1):
                snap_path = os.path.join(snapshot_dir, f"policy_snapshot_{idx:04d}.pkl")
                with open(snap_path, "wb") as f:
                    pickle.dump(snapshot, f)
            print(f"Policy snapshots saved to '{snapshot_dir}'.")

        reward_path = os.path.join(output_dir, "cr_reward.csv")
        with open(reward_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "SatisfactionRate"])
            writer.writerows(cr_reward_rows)
        print(f"CR reward log (shaped)         -> '{reward_path}'.")

        if cr_reward_cf_rows:
            cf_path = os.path.join(output_dir, "cr_reward_counterfactual_only.csv")
            with open(cf_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "SatisfactionRate"])
                writer.writerows(cr_reward_cf_rows)
            print(f"CR reward log (counterfact.)   -> '{cf_path}'.")

        if cr_reward_shaping_rows:
            sh_path = os.path.join(output_dir, "cr_reward_shaping_term.csv")
            with open(sh_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "ShapingTerm"])
                writer.writerows(cr_reward_shaping_rows)
            print(f"CR reward log (shaping term)   -> '{sh_path}'.")

        global_path = os.path.join(output_dir, "cr_reward_global.csv")
        with open(global_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "SatisfactionRate"])
            writer.writerows(cr_reward_global_rows)
        print(f"CR reward log (global)         -> '{global_path}'.")

    print(f"Done. {args.steps} steps | scenario={args.scenario} | "
          f"mode={args.mode} | strategy={args.strategy or 'RL'} | seed={args.seed}. "
          f"Logs -> '{output_dir}'.")


if __name__ == "__main__":
    main()
