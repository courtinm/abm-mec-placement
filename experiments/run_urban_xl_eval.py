import subprocess
import sys
from pathlib import Path

STRATEGIES = ["no_cr", "random", "static", "exhaustive_greedy"]
SEEDS = list(range(20))
EVAL_STEPS = 300
WARMUP = 50
DOMINANT_STATE = "2|2|2|2|2|2|2|2"


RUN_EXPERIMENT = Path(__file__).with_name("run_experiment.py")


def run(args):
    cmd = [sys.executable, str(RUN_EXPERIMENT), *args]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(variants):
    for variant in variants:
        base = Path("output") / f"output-{variant}"
        rn_qtable = base / "models/tseed0/rn"
        cr_qtable = base / "models/tseed0/cr_qtable.pkl"

        for strat in STRATEGIES:
            for seed in SEEDS:
                out = base / "logs/eval" / strat / f"s{seed}"
                summary = out / "satisfaction_summary.csv"
                if summary.exists():
                    continue
                run([
                    "--scenario", variant, "--mode", "eval", "--strategy", strat,
                    "--steps", str(EVAL_STEPS), "--seed", str(seed), "--warmup", str(WARMUP),
                    "--output-dir", str(out),
                    "--rn-qtable", str(rn_qtable),
                ])

        for seed in SEEDS:
            out = base / "logs/eval/rl" / f"s{seed}"
            summary = out / "satisfaction_summary.csv"
            if summary.exists():
                continue
            run([
                "--scenario", variant, "--mode", "eval",
                "--steps", str(EVAL_STEPS), "--seed", str(seed), "--warmup", str(WARMUP),
                "--output-dir", str(out),
                "--rn-qtable", str(rn_qtable), "--cr-qtable", str(cr_qtable),
                "--track-eval-coverage", "--dominant-state", DOMINANT_STATE,
            ])

        print(f"=== {variant} EVAL DONE ===", flush=True)


if __name__ == "__main__":
    variants = sys.argv[1:] or ["urban_xl_lambda0", "urban_xl_lambda01"]
    main(variants)
