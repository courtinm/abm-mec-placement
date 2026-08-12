"""
run_urban_xl_train.py
----------------------
Training driver for Urban-XL (dissertation §7.8.3/§7.8.4) -- the piece
run_urban_xl_eval.py / run_urban_xl_eval_tseed.py were missing. Those two are
eval-only and assume output/output-{variant}/models/tseed{0,1,2}/ already
exists; this script produces it, one RN + CR training run per (variant, train
seed), following the same ensure_train()/skip-if-exists pattern as
run_large_n12_experiments.py so it can be safely re-run without redoing
finished work.

CR_TRAIN_STEPS matches the dissertation's Urban-XL training budget (§7.8.3:
"approximately 27 000 steps", with the exploration floor raised from
epsilon_min=0.05 to 0.1 so it doesn't decay away -- already baked into
configs/urban_xl_lambda{0,01}.py's rl_hyperparams, not overridden here).

--state-coverage-every / --snapshot-every are passed during train_cr so the
resulting logs carry the state-coverage-growth and policy-snapshot data
Table 7.14 needs, not just the Q-table itself.

Usage
-----
    python experiments/run_urban_xl_train.py                    # both variants, 3 train seeds
    python experiments/run_urban_xl_train.py urban_xl_lambda0   # one variant, 3 train seeds

(always run from the project root, abm_communication_networks/)
"""

import subprocess
import sys
from pathlib import Path

TRAIN_SEEDS = [0, 1, 2]
RN_TRAIN_STEPS = 1500
CR_TRAIN_STEPS = 27000

RUN_EXPERIMENT = Path(__file__).with_name("run_experiment.py")


def run(args):
    cmd = [sys.executable, str(RUN_EXPERIMENT), *args]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_train(variant, train_seed):
    base = Path("output") / f"output-{variant}"
    model_dir = base / "models" / f"tseed{train_seed}"
    rn_base = model_dir / "rn"
    rn_model_1 = Path(f"{rn_base}_1.pkl")
    cr_model = model_dir / "cr_qtable.pkl"

    if not rn_model_1.exists():
        run([
            "--scenario", variant, "--mode", "train_rn",
            "--steps", str(RN_TRAIN_STEPS), "--seed", str(train_seed),
            "--output-dir", str(base / "logs" / f"train_rn_tseed{train_seed}"),
            "--rn-qtable", str(rn_base),
        ])
    else:
        print(f"[skip] {variant} tseed{train_seed} RN already trained ({rn_model_1})")

    if not cr_model.exists():
        run([
            "--scenario", variant, "--mode", "train_cr",
            "--steps", str(CR_TRAIN_STEPS), "--seed", str(train_seed),
            "--output-dir", str(base / "logs" / f"train_cr_tseed{train_seed}"),
            "--rn-qtable", str(rn_base), "--cr-qtable", str(cr_model),
            "--state-coverage-every", "500",
            "--snapshot-every", "1000", "--snapshot-min-state-visits", "10",
        ])
    else:
        print(f"[skip] {variant} tseed{train_seed} CR already trained ({cr_model})")


def main(variants):
    for variant in variants:
        for tseed in TRAIN_SEEDS:
            ensure_train(variant, tseed)
        print(f"=== {variant} TRAIN DONE ({len(TRAIN_SEEDS)} seeds) ===", flush=True)


if __name__ == "__main__":
    variants = sys.argv[1:] or ["urban_xl_lambda0", "urban_xl_lambda01"]
    main(variants)
