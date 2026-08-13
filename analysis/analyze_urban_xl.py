"""
analyze_urban_xl.py
--------------------
Reads the Urban-XL training/eval logs written by experiments/run_urban_xl_train.py
and experiments/run_urban_xl_eval*.py under output/output-{variant}/, and produces
Tables 7.14 (training diagnostics: state coverage, dominant-state share, Q-value
spread) and 7.15 (RL vs. baselines paired difference on post-warmup global
satisfaction) of the dissertation (Section 7.8.3/7.8.4).

Usage
-----
    python analysis/analyze_urban_xl.py

(run from the project root, abm_communication_networks/, after run_urban_xl_train.py
and run_urban_xl_eval.py / run_urban_xl_eval_tseed.py have populated output/)
"""

import csv
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stats_utils import ci95

TSEED_MAP = {
    "urban_xl_lambda0": [0, 1, 2],
    "urban_xl_lambda01": [0],
}
BASELINES = ["no_cr", "random", "static", "exhaustive_greedy"]
BASELINE_LABELS = {
    "no_cr": "No-CR", "random": "Random", "static": "Static",
    "exhaustive_greedy": "Greedy (oracle)",
}
EVAL_SEEDS = list(range(20))
WARMUP = 50
N_BS = 8
MAX_STATES = 3 ** N_BS
GROWTH_WINDOW = 5000


def rl_eval_dir(base, tseed):
    # run_urban_xl_eval.py wrote tseed0's RL eval to plain "rl"; tseed1/tseed2
    # were added later by run_urban_xl_eval_tseed.py under "rl_tseed{t}".
    name = "rl" if tseed == 0 else f"rl_tseed{tseed}"
    return base / "logs" / "eval" / name


def training_diagnostics(base, tseed):
    log_dir = base / "logs" / f"train_cr_tseed{tseed}"
    model_dir = base / "models" / f"tseed{tseed}"

    cov = pd.read_csv(log_dir / "state_coverage_checkpoints.csv")
    final_step = int(cov["step"].max())
    final_states = int(cov.loc[cov["step"] == final_step, "states_in_qtable"].iloc[0])
    pct_visited = 100 * final_states / MAX_STATES

    window_start = final_step - GROWTH_WINDOW
    prior = cov[cov["step"] <= window_start]
    if len(prior):
        prior_states = int(prior["states_in_qtable"].iloc[-1])
        growth_pct = 100 * (final_states - prior_states) / prior_states if prior_states else None
    else:
        growth_pct = None

    visits = {}
    with open(log_dir / "cr_state_visit_counts.csv") as f:
        for row in csv.DictReader(f):
            state = tuple(int(x) for x in row["state"].split("|"))
            visits[state] = int(row["visits"])
    total_visits = sum(visits.values())
    dominant_state, dominant_visits = max(visits.items(), key=lambda kv: kv[1])
    dominant_pct = 100 * dominant_visits / total_visits

    with open(model_dir / "cr_qtable.pkl", "rb") as f:
        qtable = pickle.load(f)

    def q_spread(states):
        spreads = [float(np.std(list(qtable[s].values()))) for s in states if s in qtable]
        return float(np.mean(spreads)) if spreads else None

    states_gt100 = [s for s, v in visits.items() if v > 100]
    states_lt20 = [s for s, v in visits.items() if v < 20]

    return {
        "tseed": tseed,
        "states_visited": final_states,
        "states_visited_pct": round(pct_visited, 4),
        "growth_final_5000_steps_pct": round(growth_pct, 2) if growth_pct is not None else None,
        "dominant_state": "|".join(map(str, dominant_state)),
        "dominant_state_pct_of_training": round(dominant_pct, 2),
        "states_visited_gt100": len(states_gt100),
        "std_q_states_gt100_visits": round(q_spread(states_gt100), 4) if states_gt100 else None,
        "std_q_states_lt20_visits": round(q_spread(states_lt20), 4) if states_lt20 else None,
    }


def postwarmup_global(summary_csv):
    df = pd.read_csv(summary_csv)
    df = df[df["Step"] > WARMUP]
    return float(df["Rate_Global"].mean())


def collect_global(eval_dir):
    out = {}
    for s in EVAL_SEEDS:
        p = eval_dir / f"s{s}" / "satisfaction_summary.csv"
        out[s] = postwarmup_global(p)
    return out


def dominant_hit_rate(eval_dir):
    vals = []
    for s in EVAL_SEEDS:
        p = eval_dir / f"s{s}" / "eval_state_coverage.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        vals.append(float(df["pct_dominant"].iloc[0]))
    return float(np.mean(vals)) if vals else None


def eval_summary(base, tseed):
    rl_dir = rl_eval_dir(base, tseed)
    rl_vals = collect_global(rl_dir)
    hit_rate = dominant_hit_rate(rl_dir)

    row = {
        "tseed": tseed,
        "dominant_state_hit_rate_pct": round(hit_rate, 2) if hit_rate is not None else None,
    }
    for b in BASELINES:
        b_dir = base / "logs" / "eval" / b
        b_vals = collect_global(b_dir)
        diffs = [rl_vals[s] - b_vals[s] for s in EVAL_SEEDS]
        m = float(np.mean(diffs))
        c = ci95(diffs)
        label = BASELINE_LABELS[b].lower().replace(" ", "_").replace("(", "").replace(")", "")
        row[f"rl_minus_{label}_mean_diff"] = round(m, 4)
        row[f"rl_minus_{label}_ci95"] = round(c, 4)
        row[f"rl_minus_{label}_significant"] = bool(abs(m) > c)
    return row


def main():
    out_dir = Path("results") / "Urban-XL"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_rows = []
    eval_rows = []
    for variant, tseeds in TSEED_MAP.items():
        base = Path("output") / f"output-{variant}"
        for t in tseeds:
            row = training_diagnostics(base, t)
            row["variant"] = variant
            training_rows.append(row)

            erow = eval_summary(base, t)
            erow["variant"] = variant
            eval_rows.append(erow)

    train_cols = ["variant", "tseed", "states_visited", "states_visited_pct",
                  "growth_final_5000_steps_pct", "dominant_state",
                  "dominant_state_pct_of_training", "states_visited_gt100",
                  "std_q_states_gt100_visits", "std_q_states_lt20_visits"]
    pd.DataFrame(training_rows)[train_cols].to_csv(
        out_dir / "table7_14_training_diagnostics.csv", index=False)

    eval_cols = ["variant", "tseed", "dominant_state_hit_rate_pct"]
    for b in BASELINES:
        label = BASELINE_LABELS[b].lower().replace(" ", "_").replace("(", "").replace(")", "")
        eval_cols += [f"rl_minus_{label}_mean_diff", f"rl_minus_{label}_ci95",
                      f"rl_minus_{label}_significant"]
    pd.DataFrame(eval_rows)[eval_cols].to_csv(
        out_dir / "table7_15_eval_paired_diff.csv", index=False)

    print("Written to", out_dir)


if __name__ == "__main__":
    main()
