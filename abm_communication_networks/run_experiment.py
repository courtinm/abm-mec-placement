import argparse
import importlib
import os
import random

from main import build_simulation

SCENARIOS = ("urban", "suburban", "rural", "default")

def load_config(scenario):
    module = importlib.import_module(f"configs.{scenario}")
    return module.CONFIG

def main():
    parser = argparse.ArgumentParser(description="Run headless simulation (no pygame)")
    parser.add_argument("--scenario", choices=SCENARIOS, default="urban",
                        help="Scenario configuration to use (default: urban)")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--output-dir", default=None, help="Directory for output CSV files (default: logs/<scenario>)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else os.path.join("logs", args.scenario)

    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    os.makedirs(output_dir, exist_ok=True)

    config = load_config(args.scenario)
    sim = build_simulation(config)
    for _ in range(args.steps):
        sim.simulate_step()

    sim.finalize(output_dir=output_dir)
    print(f"Done. {args.steps} steps, scenario='{args.scenario}'. Logs in '{output_dir}'.")

if __name__ == "__main__":
    main()
