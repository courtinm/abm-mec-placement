Results used in Chapter 7 of the dissertation
==============================================================

Chapter7_ExperimentA_baseline_comparison/
    experimentA_satisfaction_by_strategy_and_scenario.csv
        -> Tables 7.3 (overall satisfaction) and 7.4 (by application type),
           mean +/- 95% CI over n=20 seeds, for the 3 scenarios (Urban-Light,
           Urban-Medium, Urban-Dense) and the 5 strategies (No-CR, Random,
           Static, Greedy/oracle, RL Agent).
           Reconstructed from the raw logs (satisfaction_summary.csv per
           seed), since this table previously existed only as PNG figures
           in the pipeline; the values recalculated here correspond exactly
           to those published in the dissertation.
    {scenario}_paired_diff_n5.csv / _n20.csv
        -> Table 7.8: paired difference (seed by seed) between the RL agent
           and each baseline, with 95% CI and significance test.

Chapter7_ExperimentB_latency_efficiency/
    expB_summary.csv
        -> Table 7.5: satisfaction and CR utilization (rho) of the RL agent
           across the 3 scenarios (RQ2).

Chapter7_ExperimentC_mobility_migration/
    expC_summary.csv
        -> Table 7.6: naive comparison Urban-Light vs Urban-Medium
           (mobility confounded with load).
    expC_controlled_summary.csv
        -> Table 7.7: controlled comparison Urban-Medium vs
           Urban-Controlled (mobility isolated, load fixed at 30 users, RQ3).

Chapter7_ExperimentD_admission_allocation/
    factorial_analysis.csv
        -> Table 7.11: 2x2 factorial decomposition (cr_admission_policy
           x radio_allocation), conditions M0-M3, Section 7.7.
           'effect' column values: admission_effect, allocation_effect,
           interaction_effect (Eq. 6.4-6.6).
    factorial_analysis_full_review.csv
        -> Detailed/verification version of the same computation.

Chapter7_RN_CR_training_convergence/
    {scenario}_rn_reward.csv
        -> Table 7.1 and Figure 7.1 (top): reward of the RN agent per
           training step.
    {scenario}_cr_reward.csv / _cr_reward_global.csv
        -> Table 7.2 and Figure 7.1 (bottom): (shaped) reward of the CR
           agent per training step.
    {scenario}_cr_delta_q.csv
        -> Figure 7.2: |Delta Q| convergence of the CR agent.
    Scenarios covered: urban_light, urban_medium, urban_dense,
    urban_controlled (the latter used only in Section 7.5.2 / Table 7.7).

Chapter7_complete_data_by_scenario/
    Full copy of generate_results.py's output for the 13 scenario/condition
    combinations of the final dissertation (urban_light, urban_medium,
    urban_dense, urban_controlled, and the __M1/__M2/__M3 variants of
    light/medium/dense for Experiment D, Section 6.2.4).
    For each {scenario}/ folder:
        figures/            fig1_training_curves.png ... fig5_satisfaction_
                             timeseries.png, fig3b_paired_diff.png
        paired_diff_n5.csv / paired_diff_n20.csv   (Table 7.8, same as ExperimentA)
        logs/train_rn/      rn_reward.csv (+ rn{id}_learning.csv)
        logs/train_cr/      cr_reward.csv, cr_reward_global.csv, cr_delta_q.csv
        logs/eval/{strategy}/s{seed}/
                             raw output of one individual evaluation run,
                             pruned down to the 3 files actually read by an
                             analysis script in the repository:
                               satisfaction_summary.csv  (source of every
                                 satisfaction table: generate_results.py,
                                 paired_diff, factorial_analysis, expB/expC...)
                               cr_utilization.csv        (Table 7.5, RQ2 --
                                 read by analyze_experiment_B.py / expC*.py)
                               hop_counts.csv             (reproducibility
                                 test, Section 5.4.2)
        models/              cr_qtable.pkl, rn_{id}.pkl (final Q-tables used
                             for deterministic evaluation, Section 5.2.3)

    Removed (written by the simulator but never read back by any analysis
    script in the repository): handoffs.csv, latency_avg.csv, latency_max.csv,
    failed_connections.csv, nlos_connections.csv, optimal_connections.csv,
    satisfaction_users.csv, rn{id}_learning.csv, rn{id}_q.pkl,
    rn{id}_epsilon.pkl, and the per-seed debug logs.

    Not included: the Section 7.8 results (robustness at larger scale,
    Urban-XL/N=8 and Urban-N12, Tables 7.12-7.15). The training/evaluation
    scripts and configs for both are present in the repository and have
    been smoke-tested (see README.md, "Status of this repository", and
    EXPERIMENTS.md, Stage 7); the full-scale training runs themselves have
    not been executed, so there is no corresponding output folder here yet.
