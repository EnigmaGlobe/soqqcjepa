Source data:
D:\Soqqle\v2\soqqcjepa\testdata\1\actions_frame_train_01.csv
D:\Soqqle\v2\soqqcjepa\testdata\1\observations_frame_train_01.csv

Yes — “performance” is a defensible higher-level construct, but for your case I would name it more specifically:

PushBlock Episode Performance Quality
or, if the goal is selecting training data:

Trajectory Quality for PushBlock Training

That wording fits both the RL literature and the CFA/SEM logic.

What I found from studies

In RL and robot-learning studies, researchers usually do not use CFA to choose training data. They more often use trajectory-level performance indicators, such as return/reward, success rate, task completion, or benchmark score. For example, D4RL treats offline RL as learning from static datasets and highlights dataset types such as human demonstrations, hand-designed controllers, multitask data, and mixtures of policies. That is very close to your problem: deciding which collected data are useful for training.

There is also direct support for the idea that not all trajectories are equally useful. Offline RL work on imbalanced datasets argues that when a dataset is dominated by suboptimal trajectories, algorithms may be forced to imitate weak behaviour; they therefore propose sampling strategies that focus more on “good data.” Another paper says the target policy’s performance is strongly related to the trajectory return distribution of the dataset and proposes reweighting mixed datasets toward higher-return behaviour.

So your general idea is strong: you are not just measuring the agent; you are trying to identify which episodes/trajectories are high-quality training examples.

Where CFA fits better

CFA is more common in simulation training, VR training, and psychometric-style assessment than in standard RL benchmarking. One VR training SEM study created a latent variable called user simulation performance, measured from objective behavioural indicators: time on task, error count, question count, and fixation on a checklist. The same paper reports that CFA/SEM was used to test the hypothesised model, with fit indices such as CFI and RMSEA.

Another VR assessment study used CFA to build a four-factor model from VR-based tests, with domains such as Balance and Gait, Decision-Making, Manual Dexterity, and Memory, and then generated composite domain scores. This supports your approach: using behavioural data to create higher-level constructs is acceptable, as long as the construct is clearly defined before testing.

Best construct for your case

I would use this as the main construct:

PushBlock Episode Performance Quality

Definition:

The degree to which an episode moves the block toward the goal, preserves useful progress, and does so efficiently enough to be useful as training data.

This is better than just calling it “performance,” because it tells the reader what performance means in PushBlock.

I would break it into three subconstructs:

Higher-level construct Meaning Possible indicators
Task Achievement Did the block reach or approach the goal? neg_final_goal_distance, neg_best_goal_distance, final_progress_ratio
Progress Retention Did the episode preserve useful progress instead of losing it? neg_goal_regression, near_goal_fraction, stay_near_after_best
Execution Efficiency Did the episode achieve progress with fewer wasted steps/time? neg_episode_steps, log_progress_rate, reward_rate

This gives you a stronger theoretical story than one flat “performance” factor.

CFA model I would ask the other AI to test

Try this first as a three-factor correlated model:

TaskAchievement =~
neg_final_goal_distance +
neg_best_goal_distance +
final_progress_ratio

ProgressRetention =~
neg_goal_regression +
near_goal_fraction +
stay_near_after_best

ExecutionEfficiency =~
neg_episode_steps +
log_progress_rate +
reward_rate

TaskAchievement ~~ ProgressRetention
TaskAchievement ~~ ExecutionEfficiency
ProgressRetention ~~ ExecutionEfficiency

Then, if that works, try a higher-order performance model:

TaskAchievement =~
neg_final_goal_distance +
neg_best_goal_distance +
final_progress_ratio

ProgressRetention =~
neg_goal_regression +
near_goal_fraction +
stay_near_after_best

ExecutionEfficiency =~
neg_episode_steps +
log_progress_rate +
reward_rate

PushBlockPerformance =~
TaskAchievement +
ProgressRetention +
ExecutionEfficiency

This is the most “higher-level construct” version.

My recommendation

Use PushBlock Episode Performance Quality as the main construct.

Use these three dimensions:

Task Achievement
Progress Retention
Execution Efficiency

Then rank episodes using the CFA factor score or a validated composite score.

One important caution: CFA assumes the indicators reflect an underlying construct. But in your case, some indicators are mathematically related, such as final distance and final progress ratio. That means you should present this carefully as a validated episode-quality score, not as a deep psychological latent trait. CFA best-practice guidance also emphasizes that the measurement model should be selected using prior validity evidence, not only because it gives good fit.

A clean sentence for your paper/report could be:

Episode-level PushBlock performance quality was modelled as a higher-order construct reflecting task achievement, progress retention, and execution efficiency. This construct was used to identify trajectories that were more suitable for subsequent training, rather than relying on reward alone.

give me the proposed calculation for the metrics for the variables
Thought for 25s

Yes — below is the proposed calculation plan for your CFA variables. The key idea is: first convert your raw PushBlock frame/step data into episode-level metrics, then use those metrics as CFA indicators. Your raw data has episode_id, step_index, reward, agent position, block position, block velocity, and goal position, so it is suitable for this episode-level transformation.

0. First create the base distance variable

Use block-to-goal distance, not agent-to-goal distance, because PushBlock performance depends on moving the block to the goal.

For each row/frame:

block_goal_distance =
sqrt(
(block_pos_x - goal_pos_x)^2 +
(block_pos_z - goal_pos_z)^2
)

Use the x-z plane because this is a ground-plane task. I would ignore y unless your block can move vertically.

Also calculate:

block_speed =
sqrt(block_vel_x^2 + block_vel_z^2)

Then group by:

run_id + episode_id

Each episode becomes one row.

Construct 1: Task Achievement

This construct means:

Did the episode move the block close to the goal?

These are the strongest CFA indicators for the main “performance” construct.

1. start_goal_distance

Distance from block to goal at the beginning of the episode.

start_goal_distance = first(block_goal_distance)

Use the first valid row after removing the init row.

2. final_goal_distance

Distance from block to goal at the end of the episode.

final_goal_distance = last(block_goal_distance)

Lower is better, so for CFA use:

neg_final_goal_distance = -final_goal_distance

Interpretation:

Higher value = block ended closer to the goal.

3. best_goal_distance

Closest the block ever got to the goal during the episode.

best_goal_distance = min(block_goal_distance)

For CFA:

neg_best_goal_distance = -best_goal_distance

Interpretation:

Higher value = block reached a better position at some point.

4. final_progress_ratio

How much progress was preserved by the end.

final_progress_ratio =
(start_goal_distance - final_goal_distance) / start_goal_distance

Interpretation:

Higher value = the block finished closer to the goal than where it started.

This is very important because an episode may reach a good position briefly but then lose it.

5. best_progress_ratio

How much progress was achieved at the best moment.

best_progress_ratio =
(start_goal_distance - best_goal_distance) / start_goal_distance

Interpretation:

Higher value = the episode at least achieved strong progress at some point.

This is similar to your earlier normalized_task_progress, which appears to measure best achieved progress rather than final retained progress.

Construct 2: Progress Retention / Stability

This construct means:

After the block reached its best position, did the episode keep that progress?

This is useful because good training episodes should not only touch a good state briefly; they should preserve useful progress.

6. goal_regression

How much progress was lost after the best moment.

goal_regression =
final_goal_distance - best_goal_distance

Lower is better. For CFA:

neg_goal_regression = -goal_regression

Interpretation:

Higher value = the block did not drift far away after its best position.

This was also one of the variables previously suggested for the PushBlock performance bundle.

7. near_goal_fraction

Proportion of episode frames where the block was near the goal.

near_goal_fraction =
number_of_rows_where(block_goal_distance <= near_goal_threshold) /
total_number_of_rows

Suggested threshold:

near_goal_threshold = 1.0

But if Unity has an official success threshold, use that instead.

Interpretation:

Higher value = the block spent more time near the goal.

Caution: this may only work well for stronger episodes. In weak episodes, the block may never get near the goal, so this variable can become mostly zero.

8. stay_near_after_best

Whether the block stayed near the goal after reaching its best position.

First identify:

best_step =
step_index where block_goal_distance is minimum

Then calculate:

stay_near_after_best =
number_of_rows_after_best_where(block_goal_distance <= near_goal_threshold) /
number_of_rows_after_best

Interpretation:

Higher value = the episode maintained near-goal behaviour after reaching the best state.

This is a good secondary stability metric, but I would not make it the main indicator unless it behaves well in both top and bottom episode groups.

9. final_20pct_mean_distance

Mean block-goal distance during the final 20% of the episode.

final_20pct_mean_distance =
mean(block_goal_distance during final 20% of rows)

For CFA:

neg_final_20pct_mean_distance =
-final_20pct_mean_distance

Interpretation:

Higher value = the block stayed closer to the goal near the end.

This may be more stable than using only the final frame.

Construct 3: Execution Efficiency

This construct means:

Did the episode achieve progress efficiently?

This is useful if you want to select better training data, not merely episodes where the block eventually reached the goal.

10. episode_steps

Total number of steps in the episode.

episode_steps = max(step_index) - min(step_index) + 1

For CFA:

neg_episode_steps = -episode_steps

Interpretation:

Higher value = fewer steps used.

If all episodes have the same maximum length, this may not be useful.

11. episode_time

Elapsed real time for the episode.

episode_time =
last(realtime_since_start) - first(realtime_since_start)

For CFA:

neg_episode_time = -episode_time

Interpretation:

Higher value = faster episode.

Use either episode_steps or episode_time, not necessarily both, because they may measure nearly the same thing.

12. progress_rate

Progress per step.

progress_rate =
final_progress_ratio / episode_steps

Because this can be very small, use:

log_progress_rate =
log(final_progress_ratio + 1e-6) - log(episode_steps)

Or simpler:

log_progress_rate =
log((final_progress_ratio + 1e-6) / episode_steps)

Interpretation:

Higher value = more progress per step.

For CFA, this is useful because it adds an efficiency dimension beyond distance alone.

13. block_path_length

Total distance travelled by the block.

For each episode:

block_path_length =
sum(
sqrt(
(block_pos_x_t - block_pos_x_t-1)^2 +
(block_pos_z_t - block_pos_z_t-1)^2
)
)

Interpretation:

How much the block moved overall.

This alone is not “better” or “worse.” It becomes useful when combined with directness.

14. block_movement_directness

How directly the block movement reduced the goal distance.

block_movement_directness =
(start_goal_distance - final_goal_distance) / block_path_length

Add protection:

if block_path_length == 0, set as 0 or NA

Interpretation:

Higher value = the block’s movement was useful rather than wasted.

This is a good efficiency indicator.

Construct 4: Agent–Block Interaction Quality

This is optional, but potentially useful for training data selection.

It means:

Did the agent actually engage with the block in a useful way?

15. agent_block_distance

For each row:

agent_block_distance =
sqrt(
(agent_pos_x - block_pos_x)^2 +
(agent_pos_z - block_pos_z)^2
) 16. engagement_fraction

Proportion of rows where the agent was close to the block.

engaged =
agent_block_distance <= contact_threshold

Suggested starting threshold:

contact_threshold = 1.0 or 1.25

Then:

engagement_fraction =
number_of_engaged_rows / total_rows

Interpretation:

Higher value = the agent spent more time close enough to push/control the block.

17. active_push_fraction

Proportion of rows where the agent was close to the block and the block was moving.

active_push =
(agent_block_distance <= contact_threshold) AND
(block_speed > speed_threshold)

Suggested:

speed_threshold = 0.01

Then:

active_push_fraction =
number_of_active_push_rows / total_rows

Interpretation:

Higher value = the agent was likely producing useful block movement.

Construct 5: Reward / Termination Efficiency

This should be secondary, not the main CFA construct, because reward is partly built into the environment rules. But it is still useful for checking whether the performance construct agrees with the training signal.

18. reward_sum
    reward_sum = sum(reward)

Interpretation:

Higher value = better total reward.

19. reward_rate
    reward_rate = reward_sum / episode_steps

Interpretation:

Higher value = more reward per step.

20. max_reward
    max_reward = max(reward)

This can detect success events if success gives a large reward. But do not rely on it if it becomes ceilinged in the top group.

21. success_proxy

Only use this if your environment gives a clear success reward.

Example:

success_proxy =
1 if max_reward >= success_reward_threshold
0 otherwise

For example, if success reward is 1.0:

success_proxy = max_reward >= 1.0

But I would not put success_proxy inside the main CFA if many top episodes are already all successful.

My recommended CFA variable set

For your main CFA, I would start with this:

PushBlock Performance Quality

Task Achievement:

- neg_final_goal_distance
- neg_best_goal_distance
- final_progress_ratio

Progress Retention:

- neg_goal_regression
- neg_final_20pct_mean_distance
- near_goal_fraction

Execution Efficiency:

- neg_episode_steps
- log_progress_rate
- block_movement_directness

Then test this model:

TaskAchievement =~
neg_final_goal_distance +
neg_best_goal_distance +
final_progress_ratio

ProgressRetention =~
neg_goal_regression +
neg_final_20pct_mean_distance +
near_goal_fraction

ExecutionEfficiency =~
neg_episode_steps +
log_progress_rate +
block_movement_directness

PushBlockPerformance =~
TaskAchievement +
ProgressRetention +
ExecutionEfficiency
Simpler version if the CFA becomes unstable

Use one general factor:

PushBlockPerformance =~
neg_final_goal_distance +
neg_best_goal_distance +
final_progress_ratio +
neg_goal_regression +
log_progress_rate

This is probably the safest first model.

Variables I would avoid as main CFA indicators

I would avoid these in the main model:

raw agent_pos_x / agent_pos_z
raw block_pos_x / block_pos_z
raw action_x / action_y
raw reward at each row
success_proxy alone
max_reward alone

They are either too raw, too dependent on the environment design, or not episode-level enough.

The cleanest claim is:

PushBlock performance quality was calculated from episode-level indicators of task achievement, progress retention, and execution efficiency. Higher scores indicate episodes where the block ended closer to the goal, achieved meaningful progress, retained that progress, and did so efficiently.
