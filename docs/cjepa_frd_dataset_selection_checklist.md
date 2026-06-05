**C-JEPA Dataset Selection Checklist**

Purpose
- Use this file as a working checklist while running the dataset-selection process defined in `docs/cjepa_frd_dataset_selection.md`.
- Mark each item when it is completed so progress is visible.

Project inputs
- [x] Confirm the official observation source file is `C:\soqqle\soqqcjepa\testdata\1\data_training\observations_frame_train_01 - Copy.csv`
- [x] Confirm episode folders exist in `C:\soqqle\soqqcjepa\testdata\1\episodes`
- [x] Confirm the PushBlock metric reference docs are available in `C:\soqqle\ml-agents\Project\Assets\ML-Agents\Examples\PushBlock\Docs`

Metric design
- [x] Lock the five selected metrics for this study
- [x] Confirm the direction of each metric is correct
- [x] Confirm the episode-level formulas are written clearly in the FRD
- [x] Confirm exact 50% / 50% split rule for every phase
- [x] Confirm the random baseline design

Episode-level table
- [x] Load `observations_frame_train_01 - Copy.csv`
- [x] Remove initialization-only rows or episode `0` if needed
- [x] Compute per-row `goal_distance`
- [x] Aggregate one row per episode
- [x] Compute `start_goal_distance`
- [x] Compute `best_goal_distance`
- [x] Compute `final_goal_distance`
- [x] Compute `episode_steps`
- [x] Compute `max_reward`
- [x] Compute `reward_sum`
- [x] Compute `success_proxy`
- [x] Compute `normalized_task_progress`
- [x] Compute `progress_rate`
- [x] Save `outputs/dataset_selection/episode_level_metrics.csv`

Phase 1: Time split
- [x] Rank episodes by `episode_id`
- [x] Create Group A = first 50%
- [x] Create Group B = second 50%
- [x] Calculate all five metrics for both groups
- [x] Calculate effect sizes and A vs B differences
- [x] Calculate alpha and CFI
- [x] Save Phase 1 results

Phase 2: Reward split
- [x] Rank episodes by `max_reward`
- [x] Create Group A = top 50%
- [x] Create Group B = bottom 50%
- [x] Calculate all five metrics for both groups
- [x] Calculate effect sizes and A vs B differences
- [x] Calculate alpha and CFI
- [x] Save Phase 2 results

Phase 3: Distance split
- [x] Rank episodes by `best_goal_distance` or chosen distance metric
- [x] Confirm lower distance is treated as better
- [x] Create Group A = top 50%
- [x] Create Group B = bottom 50%
- [x] Calculate all five metrics for both groups
- [x] Calculate effect sizes and A vs B differences
- [x] Calculate alpha and CFI
- [x] Save Phase 3 results

Phase 4A: Time + Reward
- [x] Standardize both IVs
- [x] Build combined score
- [x] Split exact 50% / 50%
- [x] Calculate all five metrics
- [x] Calculate alpha and CFI
- [x] Save Phase 4A results

Phase 4B: Time + Distance
- [x] Standardize both IVs
- [x] Reverse-score distance if needed
- [x] Build combined score
- [x] Split exact 50% / 50%
- [x] Calculate all five metrics
- [x] Calculate alpha and CFI
- [x] Save Phase 4B results

Phase 4C: Reward + Distance
- [x] Standardize both IVs
- [x] Reverse-score distance if needed
- [x] Build combined score
- [x] Split exact 50% / 50%
- [x] Calculate all five metrics
- [x] Calculate alpha and CFI
- [x] Save Phase 4C results

Phase 4D: Time + Reward + Distance
- [x] Standardize all three IVs
- [x] Reverse-score distance if needed
- [x] Build combined score
- [x] Split exact 50% / 50%
- [x] Calculate all five metrics
- [x] Calculate alpha and CFI
- [x] Save Phase 4D results

Phase 5: Random baseline
- [x] Create Random Set A and Random Set B
- [x] Confirm exact 50% / 50%
- [x] Calculate all five metrics
- [x] Calculate alpha and CFI
- [x] Save one random-baseline result
- [x] Optional: repeat random baseline 30 times
- [ ] Optional: summarize mean and SD across random trials

Cross-phase comparison
- [x] Build `outputs/dataset_selection/phase_comparison_summary.csv`
- [x] Build `outputs/dataset_selection/reliability_fit_summary.csv`
- [x] Compare all phases on the five metrics
- [x] Compare all phases on alpha
- [x] Compare all phases on CFI
- [x] Compare all phases against the random baseline

Decision and reporting
- [x] Identify the strongest candidate split
- [x] Check that the selected split is not chosen from reward alone
- [x] Confirm the selected split is strong on progress and goal distance too
- [x] Confirm the selected split has acceptable reliability
- [x] Write `outputs/dataset_selection/final_dataset_recommendation.md`
- [x] Add a plain-English conclusion for the paper

Optional paper figures
- [ ] Boxplots for all five metrics by phase
- [ ] Radar chart comparing candidate splits
- [ ] Bar chart for alpha and CFI by phase
- [ ] Table for the final manuscript appendix

Completion check
- [x] FRD is complete
- [ ] Checklist is complete
- [x] Episode-level table is complete
- [x] Phase summaries are complete
- [x] Reliability and fit summaries are complete
- [x] Final recommended dataset subset is complete
