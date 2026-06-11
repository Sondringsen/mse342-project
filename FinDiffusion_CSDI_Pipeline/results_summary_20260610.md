# FinDiffusion Results Summary

Generated: 2026-06-10

## Short Version

The cleanest report story is:

> We use topology-aware losses as an inductive bias for synthetic financial data generation, then evaluate usefulness primarily through a downstream deep-hedging task. Topology and stylized-fact diagnostics are supporting evidence, not the main claim.

Main completed findings:

- Topology models beat the non-topology VTL baseline on 30-day hedging CVaR in the completed snapshot aggregate.
- Best snapshot hedging CVaR: `findiff_topo_h20_w005_evalbest_20260610_2102` at `7.63015`.
- Best completed rolling market-topology distance: `findiff_topo_h20_w002_postresume_eval_20260610_2107` at `0.173924`.
- The best topology-distance model is not the same as the best hedging model, so topology distance and hedging utility should be reported as complementary metrics.
- Do not claim full macro business-cycle generation. The defensible claim is rolling market-regime / crash-like topology over 252-day, 30-stock panels.

Important caveat:

- FinDiffusion currently produces per-ticker forecast samples. The business-cycle topology diagnostic aligns `sample_000`, `sample_001`, etc. across tickers to form pseudo-joint 30-stock panels. This is useful, but not a true joint 30-asset generative model.

## Report-Safe Claim

Suggested wording:

> Topological regularization improves downstream hedging tail-risk performance relative to the non-topology baseline in our completed experiments. Rolling topology diagnostics show that some topology-regularized models also better reproduce market-regime geometry over 252-day windows, although the model with the best topology distance is not necessarily the model with the best hedging CVaR.

Avoid:

> The model generates business cycles.

Use instead:

> The model improves reproduction of rolling market topology associated with regime shifts and crash-like structure.

## VTL Horizon Sweep

Loss profile: `vol_tail_leverage`. Lower is better for MAE, Wasserstein, vol gap, and kurtosis gap. Higher is better for coverage only insofar as it approaches 0.90, pass rate, and overall score.

| horizon | run_name                          | forecast_median_mae | forecast_coverage_90 | metric_distribution_wasserstein | analysis_vol_cluster_abs_gap | analysis_excess_kurtosis_gap | stylized_synthetic_summary_pass_rate | metric_summary_overall_score |
| ------- | --------------------------------- | ------------------- | -------------------- | ------------------------------- | ---------------------------- | ---------------------------- | ------------------------------------ | ---------------------------- |
| 1       | findiff_vtl_h1_p4_20260610_2021   | 0.011185            | 0.833333             | 0.00211364                      | 0.0311703                    | 3.61907                      | 0.5                                  | 0.995222                     |
| 15      | findiff_vtl_h15_p4_20260610_2021  | 0.0136406           | 0.741739             | 0.00327316                      | 0.00990814                   | 2.87037                      | 0.75                                 | 0.993197                     |
| 30      | findiff_vtl_h30_p4_20260610_2021  | 0.0140424           | 0.832512             | 0.00210194                      | 0.0111243                    | 2.22725                      | 0.75                                 | 0.9966                       |
| 60      | findiff_vtl_h60_p4_20260610_2021  | 0.0148479           | 0.817071             | 0.00528855                      | 0.0526705                    | 1.81806                      | 0.5                                  | 0.98128                      |
| 120     | findiff_vtl_h120_p4_20260610_2021 | 0.0235824           | 0.987222             | 0.0368643                       | 0.0745251                    | 7.64499                      | 0.5                                  | 0.96206                      |

Interpretation:

- `h30` is the best VTL horizon by overall score.
- `h1` is best for forecast MAE.
- Long horizons, especially `h120`, degrade distribution and overall realism metrics.

## Plain h1 vs VTL h1

This checks whether the extra VTL loss helps at the 1-day horizon.

| run_name                          | forecast_median_mae | forecast_coverage_90 | metric_distribution_wasserstein | analysis_vol_cluster_abs_gap | metric_summary_overall_score |
| --------------------------------- | ------------------- | -------------------- | ------------------------------- | ---------------------------- | ---------------------------- |
| findiff_vtl_h1_p4_20260610_2021   | 0.011185            | 0.833333             | 0.00211364                      | 0.0311703                    | 0.995222                     |
| findiff_plain_h1_p4_20260610_2248 | 0.0112426           | 0.820052             | 0.001749                        | 0.0285527                    | 0.995255                     |

Interpretation:

- Plain h1 is slightly better on Wasserstein, vol-cluster gap, and overall score.
- VTL h1 is slightly better on forecast MAE and 90% coverage.
- Difference is small; h1 is not the main story.

## Topology Runs: Forecast And Stylized-Fact Metrics

Topology weights shown as effective normalized weights: `w001 = 0.001`, `w002 = 0.002`, `w005 = 0.005`.

| run_name                                            | horizon | topo_weight | checkpoint       | forecast_median_mae | forecast_coverage_90 | metric_distribution_wasserstein | metric_temporal_acf_squared_mae | stylized_synthetic_summary_pass_rate | metric_summary_overall_score |
| --------------------------------------------------- | ------- | ----------- | ---------------- | ------------------- | -------------------- | ------------------------------- | ------------------------------- | ------------------------------------ | ---------------------------- |
| findiff_topo_h20_w001_evalbest_20260610_2102        | 20      | 0.001       | evalbest/current | 0.0141228           | 0.872333             | 0.00420943                      | 0.0168644                       | 0.75                                 | 0.993074                     |
| findiff_topo_h20_w001_postresume_eval_20260610_2107 | 20      | 0.001       | postresume       | 0.0139699           | 0.806952             | 0.00182795                      | 0.0156685                       | 0.75                                 | 0.99425                      |
| findiff_topo_h20_w002_evalbest_20260610_2102        | 20      | 0.002       | evalbest/current | 0.0145017           | 0.867667             | 0.00386867                      | 0.0121337                       | 0.75                                 | 0.994719                     |
| findiff_topo_h20_w002_postresume_eval_20260610_2107 | 20      | 0.002       | postresume       | 0.0141083           | 0.790571             | 0.00290859                      | 0.0262054                       | 1                                    | 0.990521                     |
| findiff_topo_h20_w005_evalbest_20260610_2102        | 20      | 0.005       | evalbest/current | 0.0140878           | 0.894952             | 0.00450834                      | 0.0128154                       | 0.75                                 | 0.994286                     |
| findiff_topo_h20_w005_postresume_eval_20260610_2107 | 20      | 0.005       | postresume       | 0.0140878           | 0.894952             | 0.00450834                      | 0.0128154                       | 0.75                                 | 0.994286                     |
| findiff_topo_h30_w001_p4_20260610_2150              | 30      | 0.001       | evalbest/current | 0.0140751           | 0.869565             | 0.002563                        | 0.0120212                       | 0.75                                 | 0.995188                     |
| findiff_topo_h30_w002_p4_20260610_2139              | 30      | 0.002       | evalbest/current | 0.0153058           | 0.889082             | 0.0111794                       | 0.0087388                       | 0.75                                 | 0.993427                     |
| findiff_topo_h30_w005_p4_20260610_2139              | 30      | 0.005       | evalbest/current | 0.0153303           | 0.955411             | 0.0134498                       | 0.00759741                      | 0.75                                 | 0.993063                     |

Interpretation:

- On h20, `w002_evalbest` has the best overall stylized/forecast score among the eval-best topology runs.
- On h20 post-resume, `w001_postresume` improves forecast MAE and Wasserstein; `w002_postresume` improves pass rate but worsens some temporal metrics.
- On h30, `w001` currently looks strongest overall; larger topology weights improve some temporal/volatility metrics but hurt distribution and MAE.
- This supports reporting topology weight as a tradeoff, not a monotonic improvement.

## Rolling Business-Cycle / Market-Regime Topology Diagnostic

This diagnostic uses aligned 30-stock panels, rolling `252`-day windows, `21`-day stride, and `25` pseudo-joint synthetic samples per run. Lower topology distance is better.

Output directory: `FinDiffusion_CSDI_Pipeline/outputs/business_cycle_topology_20260610_2312`

| run_label                                           | topology_distance_to_real_median | topology_distance_to_real_mean | beta1_proxy_area_abs_gap_median | recurrence_area_abs_gap_median | lowfreq_power_ratio_abs_gap_median | n_samples | n_windows |
| --------------------------------------------------- | -------------------------------- | ------------------------------ | ------------------------------- | ------------------------------ | ---------------------------------- | --------- | --------- |
| findiff_topo_h20_w002_postresume_eval_20260610_2107 | 0.173924                         | 0.180396                       | 0.0196581                       | 0.0167945                      | 0.0166971                          | 25        | 21        |
| findiff_topo_h20_w001_postresume_eval_20260610_2107 | 0.181487                         | 0.186731                       | 0.0214296                       | 0.0184972                      | 0.0175322                          | 25        | 21        |
| findiff_topo_h20_w001_evalbest_20260610_2102        | 0.182951                         | 0.207131                       | 0.0299343                       | 0.0272771                      | 0.0188006                          | 25        | 21        |
| findiff_vtl_h30_p4_20260610_2021                    | 0.18668                          | 0.188836                       | 0.0277587                       | 0.0252993                      | 0.0212392                          | 25        | 21        |
| findiff_topo_h20_w002_evalbest_20260610_2102        | 0.188113                         | 0.200377                       | 0.0260818                       | 0.0231838                      | 0.0189262                          | 25        | 21        |
| findiff_topo_h20_w005_evalbest_20260610_2102        | 0.190712                         | 0.195771                       | 0.0293237                       | 0.0264086                      | 0.0180406                          | 25        | 21        |
| findiff_topo_h20_w005_postresume_eval_20260610_2107 | 0.190712                         | 0.195771                       | 0.0293237                       | 0.0264086                      | 0.0180406                          | 25        | 21        |
| findiff_topo_h30_w002_p4_20260610_2139              | 0.196889                         | 0.212729                       | 0.0334342                       | 0.0302869                      | 0.0176015                          | 25        | 21        |
| findiff_topo_h30_w005_p4_20260610_2139              | 0.209263                         | 0.225771                       | 0.0341222                       | 0.0308544                      | 0.0178779                          | 25        | 21        |

Interpretation:

- `h20_w002_postresume` is the best completed model by rolling topology distance.
- Several h20 topology variants beat the VTL h30 baseline on rolling topology distance.
- h30 topology variants completed so far do not beat h20 topology or VTL h30 on this diagnostic.
- This is a rolling market-regime topology test, not proof of multi-year macro business-cycle generation.

## Snapshot Hedging Aggregate

Downstream task: train a deep hedging agent on synthetic 30-day return windows and evaluate on the same real held-out 30-day windows. Lower CVaR is better; higher mean PnL and profitability are better.

Output directory: `FinDiffusion_CSDI_Pipeline/outputs/hedging_snapshot_topo_vtl_aggregate_20260610_2231`

| label                                        | mean_pnl_mean | std_pnl_mean | pct_profitable_mean | cvar_95_mean | mean_max_drawdown_mean | worst_max_drawdown_mean | best_train_epoch_mean |
| -------------------------------------------- | ------------- | ------------ | ------------------- | ------------ | ---------------------- | ----------------------- | --------------------- |
| findiff_topo_h20_w001_evalbest_20260610_2102 | -1.88039      | 2.35793      | 0.24058             | 7.82891      | 4.1674                 | 31.4476                 | 197.667               |
| findiff_topo_h20_w002_evalbest_20260610_2102 | -1.8904       | 2.31982      | 0.235749            | 7.78045      | 4.10854                | 32.08                   | 229                   |
| findiff_topo_h20_w005_evalbest_20260610_2102 | -1.8761       | 2.31263      | 0.236715            | 7.63015      | 4.07368                | 30.2873                 | 207.667               |
| findiff_vtl_h30_p4_20260610_2021             | -1.9629       | 2.64727      | 0.230918            | 8.99804      | 3.95646                | 22.6214                 | 234.333               |
| no_hedge                                     | -2.30409      | 5.91596      | 0                   | 22.4888      | 0                      | 0                       |                       |

Interpretation:

- All h20 topology eval-best models improve CVaR over VTL h30.
- Best completed snapshot hedging model is `h20_w005_evalbest`.
- The no-hedge baseline has much worse CVaR.

## Snapshot Topology + Hedging Join

Output directory: `FinDiffusion_CSDI_Pipeline/outputs/topology_hedging_snapshot_20260610_2312`

| run_label                                    | topology_distance_to_real_median | cvar_95 | mean_pnl | pct_profitable | std_pnl |
| -------------------------------------------- | -------------------------------- | ------- | -------- | -------------- | ------- |
| findiff_topo_h20_w001_evalbest_20260610_2102 | 0.182951                         | 7.82891 | -1.88039 | 0.24058        | 2.35793 |
| findiff_vtl_h30_p4_20260610_2021             | 0.18668                          | 8.99804 | -1.9629  | 0.230918       | 2.64727 |
| findiff_topo_h20_w002_evalbest_20260610_2102 | 0.188113                         | 7.78045 | -1.8904  | 0.235749       | 2.31982 |
| findiff_topo_h20_w005_evalbest_20260610_2102 | 0.190712                         | 7.63015 | -1.8761  | 0.236715       | 2.31263 |

Interpretation:

- Topology models beat VTL h30 on hedging CVaR.
- Topology distance and hedging CVaR are not monotonic.
- Best topology among snapshot hedging runs is `h20_w001_evalbest`.
- Best hedging among snapshot runs is `h20_w005_evalbest`.

## Post-Resume Hedging: Single Completed Seed

The default seed post-resume hedging run completed. The extra two-seed packed post-resume job hit the Slurm time limit before finishing all topology models, so the multi-seed post-resume aggregate is not available yet.

Output directory: `FinDiffusion_CSDI_Pipeline/outputs/hedging_postresume_topo_20260610_2125`

| label                                               | mean_pnl | std_pnl | pct_profitable | cvar_95 | mean_max_drawdown | worst_max_drawdown | best_train_epoch |
| --------------------------------------------------- | -------- | ------- | -------------- | ------- | ----------------- | ------------------ | ---------------- |
| findiff_topo_h20_w005_postresume_eval_20260610_2107 | -1.85696 | 2.33741 | 0.255072       | 7.51457 | 4.2075            | 30.3686            | 176              |
| findiff_topo_h20_w001_postresume_eval_20260610_2107 | -1.89684 | 2.18437 | 0.185507       | 7.82483 | 3.80876           | 31.1697            | 198              |
| findiff_topo_h20_w002_postresume_eval_20260610_2107 | -1.89585 | 2.31087 | 0.189855       | 8.23253 | 3.81467           | 23.1022            | 249              |
| findiff_vtl_h30_p4_20260610_2021                    | -1.94793 | 2.58966 | 0.234783       | 8.73715 | 3.93496           | 21.3221            | 249              |

Interpretation:

- In this single seed, `h20_w005_postresume` is best by CVaR and mean PnL.
- `h20_w001_postresume` also beats VTL on CVaR.
- `h20_w002_postresume` still beats VTL on CVaR but is worse than w001/w005 for hedging in this seed.
- Because this is one seed, treat as supporting evidence until the multi-seed aggregate is rerun.

## Current Incomplete / Pending Items

As of the latest status check:

- `hedge_postresume_pack_20260610_2235` was cancelled by Slurm time limit at 2026-06-10 23:36:41.
- `hedge_postresume_agg_20260610_2235` is pending with `DependencyNeverSatisfied`.
- `join_topo_hedge_post_20260610_2312` is pending on the failed aggregate, so it will not run unless resubmitted.
- h60 topology chains are still running/pending and are not included in completed result tables.
- h30/h60 all-topology comparisons depending on pending h60 summaries are not complete.

## What To Tell The Project Partner

Tell them:

> We should make hedging the primary result. The strongest current evidence is that topology-regularized FinDiffusion improves 30-day hedging CVaR versus the VTL baseline. We also added a rolling 252-day, 30-stock topology diagnostic as supporting evidence that the generated data preserves more market-regime structure. We should not claim full business-cycle generation, because real business cycles are multi-year and the current model outputs are short-horizon pseudo-joint panels.

Short numerical summary:

| Claim | Evidence |
| ----- | -------- |
| Best completed hedging model | `h20_w005_evalbest`, CVaR `7.63015` |
| VTL hedging baseline | `vtl_h30`, CVaR `8.99804` |
| Best rolling topology model | `h20_w002_postresume`, topology distance `0.173924` |
| Best completed h30 topology model | `h30_w001`, overall score `0.995188` |
| Plain vs VTL h1 | Nearly tied; plain h1 slightly higher overall score |

## Suggested Report Framing

1. Motivation: topology captures global/rolling market geometry that standard pointwise losses miss.
2. Main downstream metric: deep hedging CVaR on real held-out 30-day windows.
3. Supporting diagnostics: stylized facts, volatility clustering, rolling market topology.
4. Result: topology loss improves hedging CVaR over VTL baseline in completed experiments.
5. Nuance: best topology distance and best hedging CVaR occur in different variants, so topology is a useful inductive bias but not the sole model-selection metric.
6. Limitation: current FinDiffusion samples are pseudo-joint across tickers and short-horizon, so we evaluate rolling market-regime topology rather than full macro business cycles.

