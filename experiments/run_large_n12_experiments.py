import csv
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SCENARIO = "urban_large_N12"
TRAIN_SEEDS = [0, 1, 2]
EVAL_SEEDS = list(range(20))
RN_TRAIN_STEPS = 1500
CR_TRAIN_STEPS = 4000
EVAL_STEPS = 300
WARMUP = 50

CONDITIONS = ["fcfs", "priority"]
BASELINES = ["no_cr", "random", "static", "exhaustive_greedy"]

ROOT = Path("output")
OUT = ROOT / f"output-{SCENARIO}"

STRATEGY_COLORS = {
    "no_cr": "#95a5a6",
    "random": "#e74c3c",
    "static": "#f39c12",
    "exhaustive_greedy": "#27ae60",
    "rl": "#2980b9",
}
STRATEGY_LABELS = {
    "no_cr": "No-CR",
    "random": "Random",
    "static": "Static",
    "exhaustive_greedy": "Greedy (oracle)",
    "rl": "RL Agent",
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 100})


def ci95(values):
    n = len(values)
    if n < 2:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / math.sqrt(n)


RUN_EXPERIMENT = Path(__file__).with_name("run_experiment.py")


def run_cmd(args):
    cmd = [sys.executable, str(RUN_EXPERIMENT), *args]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def condition_dir(condition):
    return OUT / condition


def model_dir(condition, train_seed):
    return condition_dir(condition) / "models" / f"tseed{train_seed}"


def rn_qtable_base(condition, train_seed):
    return model_dir(condition, train_seed) / "rn"


def cr_qtable_path(condition, train_seed):
    return model_dir(condition, train_seed) / "cr_qtable.pkl"


def train_logs_dir(condition, train_seed, mode):
    return condition_dir(condition) / "logs" / f"{mode}_tseed{train_seed}"


def eval_dir(condition):
    return condition_dir(condition) / "logs" / "eval"


def ensure_train(condition, train_seed):
    mdl = model_dir(condition, train_seed)
    mdl.mkdir(parents=True, exist_ok=True)

    rn_base = rn_qtable_base(condition, train_seed)
    rn_model_1 = Path(f"{rn_base}_1.pkl")
    cr_model = cr_qtable_path(condition, train_seed)

    if not rn_model_1.exists():
        run_cmd([
            "--scenario", SCENARIO,
            "--mode", "train_rn",
            "--steps", str(RN_TRAIN_STEPS),
            "--seed", str(train_seed),
            "--output-dir", str(train_logs_dir(condition, train_seed, "train_rn")),
            "--rn-qtable", str(rn_base),
            "--cr-admission-policy", condition,
        ])

    if not cr_model.exists():
        run_cmd([
            "--scenario", SCENARIO,
            "--mode", "train_cr",
            "--steps", str(CR_TRAIN_STEPS),
            "--seed", str(train_seed),
            "--output-dir", str(train_logs_dir(condition, train_seed, "train_cr")),
            "--rn-qtable", str(rn_base),
            "--cr-qtable", str(cr_model),
            "--cr-admission-policy", condition,
        ])


def ensure_eval_rl(condition, train_seed, eval_seed):
    out_dir = eval_dir(condition) / f"rl_tseed{train_seed}" / f"s{eval_seed}"
    summary = out_dir / "satisfaction_summary.csv"
    if summary.exists():
        return

    run_cmd([
        "--scenario", SCENARIO,
        "--mode", "eval",
        "--steps", str(EVAL_STEPS),
        "--seed", str(eval_seed),
        "--output-dir", str(out_dir),
        "--rn-qtable", str(rn_qtable_base(condition, train_seed)),
        "--cr-qtable", str(cr_qtable_path(condition, train_seed)),
        "--cr-admission-policy", condition,
    ])


def ensure_eval_baseline(condition, strategy, eval_seed):
    out_dir = eval_dir(condition) / strategy / f"s{eval_seed}"
    summary = out_dir / "satisfaction_summary.csv"
    if summary.exists():
        return

    run_cmd([
        "--scenario", SCENARIO,
        "--mode", "eval",
        "--strategy", strategy,
        "--steps", str(EVAL_STEPS),
        "--seed", str(eval_seed),
        "--output-dir", str(out_dir),
        "--rn-qtable", str(rn_qtable_base(condition, TRAIN_SEEDS[0])),
        "--cr-admission-policy", condition,
    ])


def postwarmup_means(summary_csv):
    df = pd.read_csv(summary_csv)
    df = df[df["Step"] > WARMUP]
    return {
        "global": float(df["Rate_Global"].mean()),
        "arvr": float(df["Rate_AR_VR"].mean()),
        "streaming": float(df["Rate_Streaming"].mean()),
        "best_effort": float(df["Rate_BestEffort"].mean()),
    }


def collect_rl(condition, train_seed):
    out = {}
    for s in EVAL_SEEDS:
        p = eval_dir(condition) / f"rl_tseed{train_seed}" / f"s{s}" / "satisfaction_summary.csv"
        out[s] = postwarmup_means(p)
    return out


def collect_baseline(condition, strategy):
    out = {}
    for s in EVAL_SEEDS:
        p = eval_dir(condition) / strategy / f"s{s}" / "satisfaction_summary.csv"
        out[s] = postwarmup_means(p)
    return out


def plot_training_curves(condition, train_seed=0):
    rn_csv = train_logs_dir(condition, train_seed, "train_rn") / "rn_reward.csv"
    cr_csv = train_logs_dir(condition, train_seed, "train_cr") / "cr_reward.csv"
    if not rn_csv.exists() or not cr_csv.exists():
        return

    rn = pd.read_csv(rn_csv)
    cr = pd.read_csv(cr_csv)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle(f"Training convergence ({SCENARIO}, {condition})", fontweight="bold")

    axes[0].plot(rn["Step"], rn["TotalReward"], color="#d0e4f0", lw=1, label="Raw")
    axes[0].plot(rn["Step"], rn["TotalReward"].rolling(30, center=True, min_periods=1).mean(),
                 color="steelblue", lw=2, label="Moving average (w=30)")
    axes[0].set_title("RN Agent")
    axes[0].set_ylabel("Total reward")
    axes[0].legend(loc="upper left")

    axes[1].plot(cr["Step"], cr["SatisfactionRate"], color="#fde8c8", lw=1, label="Raw")
    axes[1].plot(cr["Step"], cr["SatisfactionRate"].rolling(30, center=True, min_periods=1).mean(),
                 color="darkorange", lw=2, label="Moving average (w=30)")
    axes[1].set_title("CR Agent")
    axes[1].set_ylabel("Counterfactual + shaping reward")
    axes[1].set_xlabel("Training step")
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(condition_dir(condition) / f"fig_training_curves_tseed{train_seed}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fcfs_bar(fcfs_rl_all, fcfs_baselines):
    # RL metric per eval seed = mean across training seeds for that eval seed.
    rl_seed_vals = []
    for s in EVAL_SEEDS:
        rl_seed_vals.append(statistics.mean(fcfs_rl_all[t][s]["global"] for t in TRAIN_SEEDS))

    rows = [{"key": "rl", "vals": rl_seed_vals}]
    for b in BASELINES:
        rows.append({"key": b, "vals": [fcfs_baselines[b][s]["global"] for s in EVAL_SEEDS]})

    labels = [STRATEGY_LABELS[r["key"]] for r in rows]
    means = [float(np.mean(r["vals"])) for r in rows]
    cis = [ci95(r["vals"]) for r in rows]
    colors = [STRATEGY_COLORS[r["key"]] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(rows))
    bars = ax.bar(x, means, yerr=cis, capsize=7, color=colors, alpha=0.87,
                  edgecolor="white", linewidth=1.2,
                  error_kw={"elinewidth": 2, "ecolor": "#333333", "capthick": 2})
    for bar, m, c in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, m + c + 0.01, f"{m:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Mean satisfaction rate (post-warmup)")
    ax.set_title(f"Strategy comparison - {SCENARIO} (FCFS)\nGlobal satisfaction +/- 95% CI", fontweight="bold")
    fig.tight_layout()
    fig.savefig(condition_dir("fcfs") / "fig_strategy_comparison_fcfs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def paired_diff_rows(rl_by_seed, baseline_by_seed):
    rows = []
    for b in BASELINES:
        diffs = []
        for s in EVAL_SEEDS:
            diffs.append(rl_by_seed[s]["global"] - baseline_by_seed[b][s]["global"])
        m = float(np.mean(diffs))
        c = ci95(diffs)
        rows.append({
            "comparison": f"RL - {STRATEGY_LABELS[b]}",
            "mean_diff": m,
            "ci95": c,
            "significant": abs(m) > c,
        })
    return rows


def plot_forest(rows, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        color = "#27ae60" if r.get("significant", False) else "#7f8c8d"
        ax.errorbar(r["mean_diff"], yi, xerr=r["ci95"], fmt="o", color=color,
                    markersize=8, capsize=6, elinewidth=2.2, capthick=2.2)
    ax.axvline(0, color="black", lw=1.1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r["comparison"] for r in rows], fontsize=10)
    ax.set_xlabel("Paired difference in mean satisfaction")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    for c in CONDITIONS:
        for t in TRAIN_SEEDS:
            ensure_train(c, t)

    for c in CONDITIONS:
        for s in EVAL_SEEDS:
            for b in BASELINES:
                ensure_eval_baseline(c, b, s)
            for t in TRAIN_SEEDS:
                ensure_eval_rl(c, t, s)

    for c in CONDITIONS:
        plot_training_curves(c, train_seed=TRAIN_SEEDS[0])

    fcfs_rl = {t: collect_rl("fcfs", t) for t in TRAIN_SEEDS}
    prio_rl = {t: collect_rl("priority", t) for t in TRAIN_SEEDS}
    fcfs_baselines = {b: collect_baseline("fcfs", b) for b in BASELINES}
    prio_baselines = {b: collect_baseline("priority", b) for b in BASELINES}

    plot_fcfs_bar(fcfs_rl, fcfs_baselines)

    # RL-vs-baselines (FCFS): use RL averaged across training seeds per eval seed.
    fcfs_rl_avg = {
        s: {
            "global": statistics.mean(fcfs_rl[t][s]["global"] for t in TRAIN_SEEDS),
            "arvr": statistics.mean(fcfs_rl[t][s]["arvr"] for t in TRAIN_SEEDS),
            "streaming": statistics.mean(fcfs_rl[t][s]["streaming"] for t in TRAIN_SEEDS),
            "best_effort": statistics.mean(fcfs_rl[t][s]["best_effort"] for t in TRAIN_SEEDS),
        }
        for s in EVAL_SEEDS
    }
    rows_fcfs = paired_diff_rows(fcfs_rl_avg, fcfs_baselines)
    save_csv(condition_dir("fcfs") / "paired_diff_rl_vs_baselines_fcfs.csv", rows_fcfs,
             ["comparison", "mean_diff", "ci95", "significant"])
    plot_forest(
        rows_fcfs,
        f"Paired differences (FCFS) - {SCENARIO}\nRL vs baselines (n={len(EVAL_SEEDS)} eval seeds)",
        condition_dir("fcfs") / "fig_paired_diff_rl_vs_baselines_fcfs.png",
    )

    # RL-vs-baselines for priority (saved as table for completeness).
    prio_rl_avg = {
        s: {
            "global": statistics.mean(prio_rl[t][s]["global"] for t in TRAIN_SEEDS),
            "arvr": statistics.mean(prio_rl[t][s]["arvr"] for t in TRAIN_SEEDS),
            "streaming": statistics.mean(prio_rl[t][s]["streaming"] for t in TRAIN_SEEDS),
            "best_effort": statistics.mean(prio_rl[t][s]["best_effort"] for t in TRAIN_SEEDS),
        }
        for s in EVAL_SEEDS
    }
    rows_prio = paired_diff_rows(prio_rl_avg, prio_baselines)
    save_csv(condition_dir("priority") / "paired_diff_rl_vs_baselines_priority.csv", rows_prio,
             ["comparison", "mean_diff", "ci95", "significant"])

    # FCFS vs priority paired difference on RL, by training seed.
    fcfs_vs_prio_rows = []
    for t in TRAIN_SEEDS:
        diffs_g = [prio_rl[t][s]["global"] - fcfs_rl[t][s]["global"] for s in EVAL_SEEDS]
        diffs_a = [prio_rl[t][s]["arvr"] - fcfs_rl[t][s]["arvr"] for s in EVAL_SEEDS]
        diffs_s = [prio_rl[t][s]["streaming"] - fcfs_rl[t][s]["streaming"] for s in EVAL_SEEDS]
        diffs_b = [prio_rl[t][s]["best_effort"] - fcfs_rl[t][s]["best_effort"] for s in EVAL_SEEDS]
        fcfs_vs_prio_rows.append({
            "comparison": f"Priority - FCFS (RL, train seed {t})",
            "mean_diff": float(np.mean(diffs_g)),
            "ci95": ci95(diffs_g),
            "significant": abs(float(np.mean(diffs_g))) > ci95(diffs_g),
            "mean_diff_arvr": float(np.mean(diffs_a)),
            "mean_diff_streaming": float(np.mean(diffs_s)),
            "mean_diff_best_effort": float(np.mean(diffs_b)),
        })

    save_csv(
        OUT / "paired_diff_priority_vs_fcfs_rl.csv",
        fcfs_vs_prio_rows,
        [
            "comparison", "mean_diff", "ci95", "significant",
            "mean_diff_arvr", "mean_diff_streaming", "mean_diff_best_effort",
        ],
    )
    plot_forest(
        fcfs_vs_prio_rows,
        f"FCFS vs strict-priority on RL ({SCENARIO})\nGlobal satisfaction paired differences",
        OUT / "fig_paired_diff_priority_vs_fcfs_rl.png",
    )

    # Short machine-readable summary to reference in the write-up.
    summary_rows = []
    for name, rl_pack, bl_pack in [
        ("fcfs", fcfs_rl_avg, fcfs_baselines),
        ("priority", prio_rl_avg, prio_baselines),
    ]:
        summary_rows.append({
            "condition": name,
            "strategy": "rl",
            "mean_global": float(np.mean([rl_pack[s]["global"] for s in EVAL_SEEDS])),
            "mean_arvr": float(np.mean([rl_pack[s]["arvr"] for s in EVAL_SEEDS])),
            "mean_streaming": float(np.mean([rl_pack[s]["streaming"] for s in EVAL_SEEDS])),
            "mean_best_effort": float(np.mean([rl_pack[s]["best_effort"] for s in EVAL_SEEDS])),
            "ci95_global": ci95([rl_pack[s]["global"] for s in EVAL_SEEDS]),
        })
        for b in BASELINES:
            vals = [bl_pack[b][s]["global"] for s in EVAL_SEEDS]
            summary_rows.append({
                "condition": name,
                "strategy": b,
                "mean_global": float(np.mean(vals)),
                "mean_arvr": float(np.mean([bl_pack[b][s]["arvr"] for s in EVAL_SEEDS])),
                "mean_streaming": float(np.mean([bl_pack[b][s]["streaming"] for s in EVAL_SEEDS])),
                "mean_best_effort": float(np.mean([bl_pack[b][s]["best_effort"] for s in EVAL_SEEDS])),
                "ci95_global": ci95(vals),
            })

    save_csv(
        OUT / "summary_metrics.csv",
        summary_rows,
        ["condition", "strategy", "mean_global", "mean_arvr", "mean_streaming", "mean_best_effort", "ci95_global"],
    )

    print("Done. Outputs under:", OUT)


if __name__ == "__main__":
    main()
