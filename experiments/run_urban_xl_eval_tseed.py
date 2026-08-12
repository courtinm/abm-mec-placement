import subprocess
import sys
from pathlib import Path

SEEDS = list(range(20))
EVAL_STEPS = 300
WARMUP = 50
DOMINANT_STATE = "2|2|2|2|2|2|2|2"
VARIANT = "urban_xl_lambda0"


RUN_EXPERIMENT = Path(__file__).with_name("run_experiment.py")


def run(args):
    cmd = [sys.executable, str(RUN_EXPERIMENT), *args]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(tseed):
    base = Path("output") / f"output-{VARIANT}"
    rn_qtable = base / f"models/tseed{tseed}/rn"
    cr_qtable = base / f"models/tseed{tseed}/cr_qtable.pkl"

    for seed in SEEDS:
        out = base / "logs/eval" / f"rl_tseed{tseed}" / f"s{seed}"
        summary = out / "satisfaction_summary.csv"
        if summary.exists():
            continue
        run([
            "--scenario", VARIANT, "--mode", "eval",
            "--steps", str(EVAL_STEPS), "--seed", str(seed), "--warmup", str(WARMUP),
            "--output-dir", str(out),
            "--rn-qtable", str(rn_qtable), "--cr-qtable", str(cr_qtable),
            "--track-eval-coverage", "--dominant-state", DOMINANT_STATE,
        ])

    print(f"=== {VARIANT} rl_tseed{tseed} EVAL DONE ===", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
