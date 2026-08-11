"""
factorial_analysis.py
------------------------------
2x2 factorial analysis (dissertation Ch.7, M0-M3) isolating the effect of
each intervention -- CR admission priority and proportional radio
allocation -- and their interaction, on a per-seed paired basis (same
methodology as the paired-difference test, Sec 6.3.4/7.6).

    M0 = fcfs      + equal_share    (baseline, existing results)
    M1 = priority  + equal_share
    M2 = fcfs      + proportional
    M3 = priority  + proportional

For each seed i (0..19), given the four paired per-seed means M0_i..M3_i for
a given (scenario, strategy, app) combination:

    effet_priorite_i    = ((M1_i - M0_i) + (M3_i - M2_i)) / 2
    effet_debit_i       = ((M2_i - M0_i) + (M3_i - M1_i)) / 2
    effet_interaction_i = (M3_i - M2_i) - (M1_i - M0_i)
                         = (M3_i - M1_i) - (M2_i - M0_i)   -- equivalent form

These are the standard main-effect / interaction contrasts for a 2x2
factorial design; effet_priorite is literally the average of the two
"priority" simple effects ((M1-M0) at equal_share, (M3-M2) at proportional),
and effet_interaction is literally their difference -- so a small
interaction is exactly the condition under which those two simple effects
agree. Reported side by side below as a visual sanity check before trusting
the formal interaction estimate: if they diverge sharply, that IS the
interaction (not independent confirmation of it, but a legibility aid).

Each effect series is then treated exactly like the paired difference d_i in
Eq. 6.3: mean +/- 1.96 * sigma / sqrt(n).

Usage
-----
    python factorial_analysis.py --scenarios urban_light --strategies trained
    python factorial_analysis.py                                   # all 3 scenarios, all 5 strategies

Requires eval logs for all 4 conditions to already exist:
    M0: resultats/resultats-{scenario}/logs/eval/{strategy}/s{seed}/satisfaction_summary.csv
    M1/M2/M3: resultats/resultats-{scenario}__M{1,2,3}/logs/eval/{strategy}/s{seed}/...
(the __M1/__M2/__M3 suffix is applied automatically by generate_results.py's
--cr-admission-policy/--radio-allocation flags -- see CONDITION_LABELS there.)

Outputs
-------
    factorial_analysis.csv   (long-format: scenario, strategy, app, effect, mean, ci95, n)
    Printed table per (scenario, strategy, app), including the priority
    simple-effect consistency check described above.
"""

import argparse
import os

import numpy as np
import pandas as pd

SCENARIOS_DEFAULT  = ["urban_light", "urban_medium", "urban_dense"]
STRATEGIES_DEFAULT = ["no_cr", "random", "static", "exhaustive_greedy", "trained"]
APP_COLS   = ["Rate_Global", "Rate_AR_VR", "Rate_Streaming", "Rate_BestEffort"]
APP_LABELS = {
    "Rate_Global":     "Global",
    "Rate_AR_VR":      "AR/VR",
    "Rate_Streaming":  "Streaming",
    "Rate_BestEffort": "Best-effort",
}
WARMUP = 50

CONDITIONS = ["M0", "M1", "M2", "M3"]


def _eval_dir(scenario, condition):
    suffix = "" if condition == "M0" else f"__{condition}"
    return os.path.join("resultats", f"resultats-{scenario}{suffix}", "logs", "eval")


def _seed_means(scenario, condition, strategy_key, seeds, app_col):
    means = []
    for seed in seeds:
        path = os.path.join(_eval_dir(scenario, condition), strategy_key, f"s{seed}", "satisfaction_summary.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing eval log: {path}\n"
                f"Run generate_results.py for condition {condition} / scenario {scenario} first."
            )
        df = pd.read_csv(path)
        df = df[df["Step"] > WARMUP]
        means.append(float(df[app_col].mean()))
    return means


def _ci95(values):
    n = len(values)
    return 0.0 if n < 2 else 1.96 * float(np.std(values, ddof=1)) / np.sqrt(n)


def compute_effects(scenario, strategy_key, app_col, seeds):
    m = {c: np.array(_seed_means(scenario, c, strategy_key, seeds, app_col)) for c in CONDITIONS}

    priority_at_equal  = m["M1"] - m["M0"]   # simple effect of priority, holding allocation at equal_share
    priority_at_prop   = m["M3"] - m["M2"]   # simple effect of priority, holding allocation at proportional
    allocation_at_fcfs = m["M2"] - m["M0"]   # simple effect of allocation, holding admission at fcfs
    allocation_at_prio = m["M3"] - m["M1"]   # simple effect of allocation, holding admission at priority

    effet_priorite    = (priority_at_equal + priority_at_prop) / 2
    effet_debit       = (allocation_at_fcfs + allocation_at_prio) / 2
    effet_interaction = priority_at_prop - priority_at_equal
    # equivalent form, sanity-checked once here rather than trusted blindly:
    effet_interaction_alt = allocation_at_prio - allocation_at_fcfs
    assert np.allclose(effet_interaction, effet_interaction_alt), (
        "Interaction contrast forms disagree -- formula bug, stop and check."
    )

    return {
        "priority_at_equal_share": priority_at_equal,
        "priority_at_proportional": priority_at_prop,
        "effet_priorite": effet_priorite,
        "effet_debit": effet_debit,
        "effet_interaction": effet_interaction,
    }


def summarize(series):
    return float(np.mean(series)), _ci95(series), len(series)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS_DEFAULT)
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES_DEFAULT)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    rows = []
    for scenario in args.scenarios:
        for strategy_key in args.strategies:
            print(f"\n{'=' * 78}\n  {scenario} / {strategy_key}\n{'=' * 78}")
            for app_col in APP_COLS:
                try:
                    eff = compute_effects(scenario, strategy_key, app_col, seeds)
                except FileNotFoundError as e:
                    print(f"  [SKIP] {APP_LABELS[app_col]}: {e}")
                    continue

                pe_mean, pe_ci, _ = summarize(eff["priority_at_equal_share"])
                pp_mean, pp_ci, _ = summarize(eff["priority_at_proportional"])
                consistent = abs(pe_mean - pp_mean) < (pe_ci + pp_ci)  # rough visual check, see docstring

                print(f"\n  [{APP_LABELS[app_col]}]")
                print(f"    priority simple effect @ equal_share : {pe_mean:+.4f} +/- {pe_ci:.4f}")
                print(f"    priority simple effect @ proportional: {pp_mean:+.4f} +/- {pp_ci:.4f}"
                      f"   ({'consistent' if consistent else 'DIVERGE -- expect a real interaction below'})")

                for effect_name in ["effet_priorite", "effet_debit", "effet_interaction"]:
                    mean, ci, n = summarize(eff[effect_name])
                    sig = "significant" if abs(mean) > ci else "not significant"
                    print(f"    {effect_name:<20} {mean:+.4f} +/- {ci:.4f}  ({sig})")
                    rows.append({
                        "scenario": scenario, "strategy": strategy_key, "app": APP_LABELS[app_col],
                        "effect": effect_name, "mean": round(mean, 4), "ci95": round(ci, 4), "n": n,
                    })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv("factorial_analysis.csv", index=False)
        print(f"\nSaved: factorial_analysis.csv")


if __name__ == "__main__":
    main()
