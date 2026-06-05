**C-JEPA FRD — Dataset Selection For Paper Experiments**

Purpose
- Define a structured, paper-ready process for selecting the best PushBlock episode subsets for C-JEPA training.
- Use measurable episode-quality evidence instead of manually taking an arbitrary early or late slice.
- Compare candidate dataset splits with a fixed metric framework so the final training set choice is defensible.

Official source files for this process
- Episode-level state source: `C:\soqqle\soqqcjepa\testdata\1\data_training\observations_frame_train_01 - Copy.csv`
- Episode folders already prepared: `C:\soqqle\soqqcjepa\testdata\1\episodes`
- Metric reference documents: `C:\soqqle\ml-agents\Project\Assets\ML-Agents\Examples\PushBlock\Docs`

Scope
- In scope:
  - selecting the best five episode-quality metrics for this dataset study
  - computing those metrics from the observation CSV
  - splitting episodes into exact 50% / 50% candidate groups under several independent variables (IVs)
  - evaluating each candidate split with descriptive statistics, reliability alpha, and CFI
  - deciding which split is the strongest training dataset candidate for C-JEPA
- Out of scope:
  - model training itself
  - hyperparameter tuning
  - architecture changes
  - reward redesign in Unity

Main idea
- We do not assume that "later episodes are better" or that "high reward alone is enough".
- Instead, we create several candidate dataset splits using different IVs.
- For each split, we calculate the same five quality metrics.
- Then we compare the splits using a common scoring and reliability framework.
- The best split is the one that shows the strongest and most coherent evidence of task-relevant, learnable behavior.

**1. Selected Metrics**
The PushBlock documents contain many useful measures. For this FRD, the best five metrics for C-JEPA dataset selection are the following.

1. Success Proxy Rate
- Source idea: Reliability Metrics
- Working definition for this dataset: the proportion of episodes where `max_reward >= 4.9`
- Why it matters:
  - it captures whether the episode contains near-complete task behavior
  - it is the clearest signal that the block was actually pushed into the goal area or very close to it
  - it gives a strong "expert-like behavior present" signal for representation learning

2. Normalized Task Progress
- Source idea: Learning Improvement Metrics and Block Progress Metrics
- Working definition:

```text
normalized_task_progress =
  (start_goal_distance - best_goal_distance) / start_goal_distance
```

- In this dataset, `start_goal_distance` is the first block-to-goal distance in the episode, and `best_goal_distance` is the smallest block-to-goal distance reached during the episode.
- Why it matters:
  - it works even when the episode does not fully succeed
  - it captures partial but meaningful pushing behavior
  - it is more useful than pure success/fail when building a training dataset

3. Final Goal Error
- Source idea: Block Progress Metrics
- Working definition: final block-to-goal distance at the end of the episode
- Why it matters:
  - it tells us how close the episode ended to task completion
  - low values indicate usable goal-directed behavior
  - it helps separate "almost solved" from "wandered and finished far away"

4. Progress Rate
- Source idea: Block Progress Metrics
- Working definition:

```text
progress_rate = normalized_task_progress / episode_steps
```

- Why it matters:
  - it measures how efficiently the episode converts time into task progress
  - two episodes can have similar progress, but one can be much cleaner and faster
  - efficient trajectories are more useful for learning predictive structure

5. Reward Consistency
- Source idea: Reliability Metrics
- Working definition: variability of per-episode reward inside a candidate split, measured by SD and IQR
- Why it matters:
  - a training subset should not only contain high-scoring episodes, it should also be behaviorally coherent
  - lower variability inside a "good" subset suggests the policy behavior is more stable and reusable
  - this gives us a reliability-style quality signal, not just a competence signal

Why these five were chosen
- They are all grounded in the PushBlock metric design documents.
- They can be computed from the observation CSV without requiring new Unity instrumentation.
- Together they cover:
  - task completion
  - partial progress
  - closeness to the goal
  - efficiency
  - reliability
- This combination is better for a paper than using reward alone.

**2. Episode-Level Derived Fields**
From `observations_frame_train_01 - Copy.csv`, compute the following episode-level fields first.

Per-row helper value

```text
goal_distance = sqrt(
  (block_pos_x - goal_pos_x)^2 +
  (block_pos_y - goal_pos_y)^2 +
  (block_pos_z - goal_pos_z)^2
)
```

Per-episode fields
- `episode_id`
- `start_goal_distance` = first `goal_distance`
- `best_goal_distance` = minimum `goal_distance`
- `final_goal_distance` = last `goal_distance`
- `episode_steps` = max `step_index`
- `max_reward` = max `reward`
- `reward_sum` = sum of `reward`
- `success_proxy` = `1` if `max_reward >= 4.9`, else `0`
- `normalized_task_progress`
- `progress_rate`

Direction rules
- Higher is better:
  - `success_proxy`
  - `normalized_task_progress`
  - `progress_rate`
- Lower is better:
  - `final_goal_distance`
  - reward variability inside a split

For any combined scoring stage, lower-is-better metrics must be reversed before aggregation.

**3. Pre-Analysis Data Preparation**
Step 1
- Load `observations_frame_train_01 - Copy.csv`.

Step 2
- Drop episode `0` if it is only initialization data.

Step 3
- Aggregate the raw rows into one row per episode.

Step 4
- Sort episodes by `episode_id` so time-based comparisons are reproducible.

Step 5
- Create a clean episode-level analysis table.

Required output of this step
- one CSV with one row per episode and all derived fields needed by the later phases

Recommended output file name
- `outputs/dataset_selection/episode_level_metrics.csv`

**4. Independent Variables (IVs) And Split Phases**
All comparisons must use exact 50% / 50% episode counts.

If the total number of episodes is odd:
- remove the median-ranked episode for that phase before splitting
- this keeps both groups exactly balanced

Important rule
- Every phase produces two groups:
  - Group A
  - Group B
- The same five metrics are calculated for both groups
- The same evaluation outputs are generated for every phase

**Phase 1: IV = Time**
Definition
- Rank episodes by `episode_id`

Split
- Group A = first 50% of episodes by time sequence
- Group B = second 50% of episodes by time sequence

Purpose
- check whether later experience is actually better than earlier experience
- do not assume this is true until the metrics confirm it

**Phase 2: IV = Reward**
Definition
- Rank episodes by `max_reward`

Split
- Group A = top 50% by reward
- Group B = bottom 50% by reward

Purpose
- check whether reward-based filtering creates a better training subset

**Phase 3: IV = Distance To Goal**
Definition
- Rank episodes by `best_goal_distance` or `final_goal_distance`
- smaller distance is better

Recommended primary ranking
- use `best_goal_distance` as the main ranking variable because it reflects whether the episode ever got the block near the goal area

Split
- Group A = top 50% with smaller goal distance
- Group B = bottom 50% with larger goal distance

Purpose
- check whether geometric closeness to the goal is a better selector than reward alone

**Phase 4: Combined IVs**
Combined phases must also remain exact 50% / 50%.

Method
- Standardize each IV into a z-score
- Reverse-score the variables where lower is better
- Average the chosen IV z-scores into one combined ranking score
- Rank episodes by that combined score
- Split top 50% vs bottom 50%

Phase 4A: Time + Reward
- combined score from `time_rank` and `reward_rank`

Phase 4B: Time + Distance
- combined score from `time_rank` and reversed `distance_rank`

Phase 4C: Reward + Distance
- combined score from `reward_rank` and reversed `distance_rank`

Phase 4D: Time + Reward + Distance
- combined score from all three IVs

Purpose
- test whether a mixed selection rule is stronger than any single IV alone

**Phase 5: Random Baseline**
Minimum requirement from this FRD
- create two exact 50% random sets:
  - Group A = random first set
  - Group B = random second set

Paper-quality recommendation
- repeat the random split at least 30 times
- report the mean and SD of the results across those random trials

Purpose
- give a null-style baseline
- prove that a proposed selection rule is better than a random split

**5. Evaluation Outputs For Every Phase**
For every phase and for both groups A and B, calculate the following.

Descriptive outputs
- number of episodes
- mean and SD of `success_proxy`
- mean and SD of `normalized_task_progress`
- mean and SD of `final_goal_distance`
- mean and SD of `progress_rate`
- reward SD and reward IQR

Comparison outputs
- A vs B mean difference for each metric
- standardized effect size for each metric
- direction check: whether the theoretically better group is actually better

Reliability outputs
- Cronbach's alpha across the standardized five-metric bundle
- split-half consistency if needed as a backup

Construct-fit output
- CFI from a one-factor confirmatory model treating the five metrics as indicators of latent dataset quality

Interpretation rule for CFI
- higher CFI means the five selected metrics behave more coherently as one dataset-quality construct
- lower CFI means the metric bundle does not fit together well for that split

Interpretation rule for alpha
- higher alpha means the chosen metrics are more internally consistent inside that split
- lower alpha means the split may be behaviorally mixed or noisy

Important note for the paper
- CFI and alpha do not prove that the dataset is "good" by themselves
- they show whether the selected metrics form a stable and coherent quality pattern
- they must be interpreted together with the descriptive metrics and effect sizes

**6. Decision Framework**
The preferred training dataset should satisfy all of the following.

Primary criteria
- stronger `success_proxy`
- stronger `normalized_task_progress`
- lower `final_goal_distance`
- stronger `progress_rate`
- lower reward variability

Reliability criteria
- acceptable Cronbach's alpha
- acceptable CFI

Comparison criteria
- the chosen split should outperform its comparison group on most or all of the five metrics
- the chosen split should also outperform or clearly separate from the random baseline

Tie-break rule
- if two candidate splits are close, prefer the one with:
  - better progress-based metrics over reward-only metrics
  - better reliability
  - better interpretability for the paper narrative

Recommended interpretation order
1. Check whether the split improves the three task-quality metrics:
   - `success_proxy`
   - `normalized_task_progress`
   - `final_goal_distance`
2. Check whether the split also improves efficiency:
   - `progress_rate`
3. Check whether the split is stable enough:
   - reward consistency
   - alpha
   - CFI

**7. Expected Deliverables**
This process should produce the following files.

1. Episode-level metric table
- `outputs/dataset_selection/episode_level_metrics.csv`

2. Phase comparison table
- one table with all phases and A/B results
- recommended file name:
  - `outputs/dataset_selection/phase_comparison_summary.csv`

3. Reliability and fit table
- alpha and CFI for every candidate split
- recommended file name:
  - `outputs/dataset_selection/reliability_fit_summary.csv`

4. Final recommendation note
- short plain-English conclusion naming the best candidate training subset
- recommended file name:
  - `outputs/dataset_selection/final_dataset_recommendation.md`

5. Optional figures for the paper
- boxplots by phase
- radar chart of the five metrics
- bar chart of alpha and CFI by phase

**8. Phase Summary Table**
The final report should summarize the experiment in this structure.

| Phase | IV | Group A | Group B | Main question |
|---|---|---|---|---|
| 1 | Time | first 50% | second 50% | does later experience produce better data? |
| 2 | Reward | top 50% reward | bottom 50% reward | is reward a good data selector? |
| 3 | Distance | top 50% closer-to-goal | bottom 50% farther-from-goal | is geometric success better than reward? |
| 4A | Time + Reward | top 50% combined | bottom 50% combined | does combining time and reward improve selection? |
| 4B | Time + Distance | top 50% combined | bottom 50% combined | does combining time and goal distance improve selection? |
| 4C | Reward + Distance | top 50% combined | bottom 50% combined | does combining reward and geometry improve selection? |
| 4D | Time + Reward + Distance | top 50% combined | bottom 50% combined | is the full combined rule strongest? |
| 5 | Random | random set A | random set B | does the proposed method beat chance? |

**9. Practical Notes For This Repo**
- This FRD uses only `observations_frame_train_01 - Copy.csv` as the official source.
- The already separated episode folders in `testdata\1\episodes` remain useful later if we want to physically create training subsets after the best split is identified.
- This FRD does not rely on the earlier manual exploring/well-trained split.
- The output of this FRD should be used to decide which episodes are moved or copied into the final C-JEPA training subset.

**10. Final Recommendation Policy**
The final paper should not say:
- "we used early episodes and late episodes because they looked different"

The final paper should say something closer to:
- "we evaluated candidate episode subsets using five task-quality metrics derived from PushBlock behavior, compared the subsets under time-, reward-, distance-, combined-, and random-split IVs, and selected the training subset with the strongest task progress, efficiency, and reliability profile."

End of FRD.