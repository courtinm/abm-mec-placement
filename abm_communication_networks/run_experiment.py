import argparse
import os
import random

from main import build_simulation

def main():
    parser = argparse.ArgumentParser(description="Run headless simulation (no pygame)")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--output-dir", default="logs", help="Directory for output CSV files")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    os.makedirs(args.output_dir, exist_ok=True)

    sim = build_simulation()
    for _ in range(args.steps):
        sim.simulate_step()

    sim.finalize(output_dir=args.output_dir)
    print(f"Done. {args.steps} steps completed. Logs in '{args.output_dir}'.")

if __name__ == "__main__":
    main()
