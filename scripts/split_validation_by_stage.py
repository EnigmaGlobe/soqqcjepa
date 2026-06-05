"""
split_validation_by_stage.py
=============================
Split a validation video into training-stage-based splits for C-JEPA evaluation.

Your ML-Agents video captures policy training with clear phases:
  - Random exploration (early): agent acts randomly, episodes timeout
  - Learning (mid): agent starts succeeding, episode length drops
  - Converged (late): agent solves consistently, short episodes

This script reads the observation CSV, detects episodes, computes per-episode
reward/length/success, and assigns each episode to a stage. It then writes
time-aligned slot/action/proprio/state pickles for each stage.

Usage
-----
python scripts/split_validation_by_stage.py \
    --obs_csv testdata/validation/3/data_training/observations_frame_train_03.csv \
    --act_csv testdata/validation/3/data_training/actions_frame_train_03.csv \
    --slot_pkl checkpoints/train01_slots.pkl \
    --output_dir checkpoints/validation_staged \
    --video_key train_03_recording \
    --n_stages 4

Output structure
----------------
checkpoints/validation_staged/
    train_03_stage_0_exploration.pkl   # early random policy
    train_03_stage_1_learning.pkl      # improving policy
    train_03_stage_2_converging.pkl    # mostly solving
    train_03_stage_3_converged.pkl     # expert policy

Each pickle contains {"train": {}, "val": {video_key: data}} compatible with
PushTSlotDataset and the validation script.

Stage definitions (default 4 stages)
------------------------------------
- stage_0 (exploration):   success_rate < 20%,  mean_reward < 0
- stage_1 (learning):      success_rate 20-80%, learning curve
- stage_2 (converging):    success_rate 80-95%, almost there
- stage_3 (converged):     success_rate > 95%,  expert performance

You can also use --method percentile to split by training_step percentiles.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_csv", required=True, help="Path to observations CSV")
    parser.add_argument("--act_csv", required=True, help="Path to actions CSV")
    parser.add_argument("--slot_pkl", required=True, help="Path to pre-extracted slots pickle")
    parser.add_argument("--output_dir", default="checkpoints/validation_staged")
    parser.add_argument("--video_key", default="train_03_recording",
                        help="Key used in the slot pickle for this video")
    parser.add_argument("--n_stages", type=int, default=4, choices=[3, 4, 5, 10],
                        help="Number of training stages to split into")
    parser.add_argument("--method", default="adaptive", choices=["adaptive", "percentile", "episode"],
                        help="How to define stages:")
    parser.add_argument("--min_episode_length", type=int, default=10,
                        help="Filter out episodes shorter than this")
    parser.add_argument("--smooth_window", type=int, default=50,
                        help="Window for smoothing success rate curve")
    parser.add_argument("--visualize", action="store_true",
                        help="Save a plot of the stage boundaries")
    parser.add_argument("--success_metric", choices=["reward", "improvement", "final_dist"], default="reward",
                        help="Which per-episode metric to treat as success when computing smoothed rates")
    parser.add_argument("--dist_thresh", type=float, default=1.0,
                        help="Distance threshold (for success_metric=final_dist)")
    return parser.parse_args()


def load_and_align_data(obs_csv: str, act_csv: str, slot_pkl: str, video_key: str):
    """Load observation, action, and slot data, align by frame index."""
    print(f"[1/5] Loading observation CSV: {obs_csv}")
    obs = pd.read_csv(obs_csv)
    # Filter out init row
    obs = obs[obs["agent_id"] != "init"].reset_index(drop=True)
    print(f"      Observations: {len(obs)} rows")

    print(f"[2/5] Loading action CSV: {act_csv}")
    act = pd.read_csv(act_csv)
    act = act[act["agent_id"] != "init"].reset_index(drop=True)
    print(f"      Actions: {len(act)} rows")

    print(f"[3/5] Loading slots: {slot_pkl}")
    with open(slot_pkl, "rb") as f:
        slot_data = pickle.load(f)

    # The slot pickle may have nested structure
    slots = None
    if video_key in slot_data:
        slots = slot_data[video_key]
    elif "train" in slot_data and video_key in slot_data["train"]:
        slots = slot_data["train"][video_key]
    elif "val" in slot_data and video_key in slot_data["val"]:
        slots = slot_data["val"][video_key]
    else:
        # Try to find any matching key
        for split in ["train", "val"]:
            if split in slot_data:
                for k in slot_data[split]:
                    if video_key in k or k in video_key:
                        slots = slot_data[split][k]
                        print(f"      Found slots under key: {split}/{k}")
                        break
                if slots is not None:
                    break

    if slots is None:
        raise ValueError(f"Could not find video_key '{video_key}' in slot pickle. Keys: {list(slot_data.keys())}")

    slots = np.asarray(slots)
    print(f"      Slots shape: {slots.shape}")

    # Align lengths (observations may have more rows due to duplicated decision steps)
    n_frames = min(len(obs), len(act), slots.shape[0])
    if len(obs) != n_frames or len(act) != n_frames or slots.shape[0] != n_frames:
        print(f"      [WARN] Length mismatch: obs={len(obs)}, act={len(act)}, slots={slots.shape[0]}")
        print(f"      Using minimum: {n_frames}")

    obs = obs.iloc[:n_frames].reset_index(drop=True)
    act = act.iloc[:n_frames].reset_index(drop=True)
    slots = slots[:n_frames]

    return obs, act, slots


def detect_episodes(obs: pd.DataFrame, min_length: int = 10):
    """Detect episode boundaries and compute per-episode stats."""
    # Episode boundaries: where episode_id changes
    ep_ids = obs["episode_id"].values
    boundaries = [0] + list(np.where(np.diff(ep_ids) != 0)[0] + 1) + [len(obs)]

    episodes = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        length = end - start
        if length < min_length:
            continue

        ep_obs = obs.iloc[start:end]
        ep_reward = ep_obs["reward"].sum()
        ep_success = ep_reward > 0
        ep_step = ep_obs["training_step"].max()

        # Agent movement
        agent_start = np.array([ep_obs["agent_pos_x"].iloc[0], ep_obs["agent_pos_z"].iloc[0]])
        agent_end = np.array([ep_obs["agent_pos_x"].iloc[-1], ep_obs["agent_pos_z"].iloc[-1]])
        agent_dist = np.linalg.norm(agent_end - agent_start)

        # Goal distance at start and end
        goal = np.array([ep_obs["goal_pos_x"].iloc[0], ep_obs["goal_pos_z"].iloc[0]])
        start_dist = np.linalg.norm(agent_start - goal)
        end_dist = np.linalg.norm(agent_end - goal)

        episodes.append({
            "start_frame": start,
            "end_frame": end,
            "length": length,
            "episode_id": ep_obs["episode_id"].iloc[0],
            "training_step": ep_step,
            "total_reward": ep_reward,
            "success": ep_success,
            "agent_dist": agent_dist,
            "start_goal_dist": start_dist,
            "end_goal_dist": end_dist,
            "improvement": start_dist - end_dist,  # positive = got closer to goal
        })

    return pd.DataFrame(episodes)


def assign_stages_adaptive(ep_df: pd.DataFrame, n_stages: int, smooth_window: int = 50):
    """Assign episodes to stages based on smoothed success rate curve."""
    ep_sorted = ep_df.sort_values("training_step").reset_index(drop=True)

    # Smooth success rate
    ep_sorted["success_float"] = ep_sorted["success"].astype(float)
    ep_sorted["smoothed_success"] = ep_sorted["success_float"].rolling(
        window=smooth_window, min_periods=1, center=True
    ).mean()

    # Define stage boundaries based on smoothed success rate
    if n_stages == 3:
        # exploration / learning / converged
        thresholds = [0.20, 0.80]
        labels = ["exploration", "learning", "converged"]
    elif n_stages == 4:
        # exploration / learning / converging / converged
        thresholds = [0.15, 0.50, 0.85]
        labels = ["exploration", "learning", "converging", "converged"]
    elif n_stages == 5:
        thresholds = [0.10, 0.30, 0.60, 0.90]
        labels = ["exploration", "early_learning", "mid_learning", "converging", "converged"]
    else:  # 10 stages = percentiles
        return assign_stages_percentile(ep_sorted, n_stages)

    # Find the training_step where smoothed success crosses each threshold
    stage_boundaries = [0]
    for thresh in thresholds:
        crossing = ep_sorted[ep_sorted["smoothed_success"] >= thresh]
        if len(crossing) == 0:
            # Never reaches this threshold, put boundary at end
            stage_boundaries.append(ep_sorted["training_step"].max() + 1)
        else:
            stage_boundaries.append(crossing["training_step"].iloc[0])
    stage_boundaries.append(ep_sorted["training_step"].max() + 1)

    # Assign each episode to a stage
    def get_stage(step):
        for i in range(len(stage_boundaries) - 1):
            if stage_boundaries[i] <= step < stage_boundaries[i + 1]:
                return i
        return len(stage_boundaries) - 2

    ep_sorted["stage_idx"] = ep_sorted["training_step"].apply(get_stage)
    ep_sorted["stage_name"] = ep_sorted["stage_idx"].apply(lambda x: labels[x])

    return ep_sorted, stage_boundaries, labels


def assign_stages_percentile(ep_df: pd.DataFrame, n_stages: int):
    """Assign episodes to stages by training_step percentiles."""
    ep_sorted = ep_df.sort_values("training_step").reset_index(drop=True)
    max_step = ep_sorted["training_step"].max()

    labels = [f"stage_{i}" for i in range(n_stages)]
    percentiles = np.linspace(0, 100, n_stages + 1)
    stage_boundaries = [0] + [np.percentile(ep_sorted["training_step"], p) for p in percentiles[1:-1]] + [max_step + 1]

    def get_stage(step):
        for i in range(len(stage_boundaries) - 1):
            if stage_boundaries[i] <= step < stage_boundaries[i + 1]:
                return i
        return len(stage_boundaries) - 2

    ep_sorted["stage_idx"] = ep_sorted["training_step"].apply(get_stage)
    ep_sorted["stage_name"] = ep_sorted["stage_idx"].apply(lambda x: labels[x])

    return ep_sorted, stage_boundaries, labels


def write_stage_pickles(ep_df: pd.DataFrame, obs: pd.DataFrame, act: pd.DataFrame,
                        slots: np.ndarray, output_dir: str, video_key: str,
                        stage_names: list[str]):
    """Write one pickle per stage with aligned slot/action/proprio/state data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract action and proprio arrays
    action_cols = [c for c in act.columns if c.startswith("action_")]
    actions = act[action_cols].values.astype(np.float32)

    # Proprio: agent position + rotation (exclude goal/block for now, can add later)
    proprio_cols = ["agent_pos_x", "agent_pos_y", "agent_pos_z",
                    "agent_rot_x", "agent_rot_y", "agent_rot_z"]
    proprios = obs[proprio_cols].values.astype(np.float32)

    # State: all position/velocity info
    state_cols = [c for c in obs.columns
                  if any(x in c for x in ["pos_", "vel_", "rot_"])]
    states = obs[state_cols].values.astype(np.float32)

    # Reward
    rewards = obs["reward"].values.astype(np.float32).reshape(-1, 1)

    written = []
    for stage_idx, stage_name in enumerate(stage_names):
        stage_eps = ep_df[ep_df["stage_idx"] == stage_idx]
        if len(stage_eps) == 0:
            print(f"      [SKIP] Stage {stage_name}: no episodes")
            continue

        # Collect all frames belonging to this stage
        frame_indices = []
        for _, ep in stage_eps.iterrows():
            frame_indices.extend(range(int(ep["start_frame"]), int(ep["end_frame"])))
        frame_indices = sorted(set(frame_indices))

        if len(frame_indices) == 0:
            print(f"      [SKIP] Stage {stage_name}: no frames")
            continue

        # Extract data for these frames
        stage_slots = slots[frame_indices]
        stage_actions = actions[frame_indices]
        stage_proprios = proprios[frame_indices]
        stage_states = states[frame_indices]
        stage_rewards = rewards[frame_indices]

        # Build pickle in the same format as training data
        # Note: we put everything in "val" split since this is validation data
        out_name = f"{video_key}_stage_{stage_idx}_{stage_name}"
        out_pkl = {
            "train": {},
            "val": {
                out_name: stage_slots,
            }
        }

        # Also save action/proprio/state/reward separately for flexibility
        meta_pkl = {
            "train": {},
            "val": {
                out_name: stage_actions,
            }
        }
        proprio_pkl = {
            "train": {},
            "val": {
                out_name: stage_proprios,
            }
        }
        state_pkl = {
            "train": {},
            "val": {
                out_name: stage_states,
            }
        }
        reward_pkl = {
            "train": {},
            "val": {
                out_name: stage_rewards,
            }
        }

        # Save all
        base = output_path / out_name
        with open(f"{base}_slots.pkl", "wb") as f:
            pickle.dump(out_pkl, f)
        with open(f"{base}_actions.pkl", "wb") as f:
            pickle.dump(meta_pkl, f)
        with open(f"{base}_proprio.pkl", "wb") as f:
            pickle.dump(proprio_pkl, f)
        with open(f"{base}_state.pkl", "wb") as f:
            pickle.dump(state_pkl, f)
        with open(f"{base}_reward.pkl", "wb") as f:
            pickle.dump(reward_pkl, f)

        # Save episode metadata
        meta = {
            "video_key": video_key,
            "stage_name": stage_name,
            "stage_idx": stage_idx,
            "n_episodes": len(stage_eps),
            "n_frames": len(frame_indices),
            "training_step_range": [int(stage_eps["training_step"].min()), int(stage_eps["training_step"].max())],
            "mean_reward": float(stage_eps["total_reward"].mean()),
            "success_rate": float(stage_eps["success"].mean()),
            "mean_length": float(stage_eps["length"].mean()),
            "episode_ids": stage_eps["episode_id"].tolist(),
        }
        with open(f"{base}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"      Stage {stage_name}: {len(stage_eps)} episodes, {len(frame_indices)} frames")
        print(f"        Success rate: {meta['success_rate']*100:.1f}%, Mean reward: {meta['mean_reward']:.3f}")
        written.append((stage_name, base))

    return written


def visualize_stages(ep_df: pd.DataFrame, stage_boundaries: list, labels: list,
                     output_path: str):
    """Save a plot showing stage boundaries on the training curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ep_sorted = ep_df.sort_values("training_step")

    # Plot 1: Episode reward with stage colors
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for stage_idx, stage_name in enumerate(labels):
        stage_eps = ep_sorted[ep_sorted["stage_idx"] == stage_idx]
        ax.scatter(stage_eps["training_step"], stage_eps["total_reward"],
                   alpha=0.4, s=8, color=colors[stage_idx], label=stage_name)

    for b in stage_boundaries[1:-1]:
        ax.axvline(b, color="black", linestyle="--", alpha=0.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Episode Total Reward")
    ax.set_title("Stage Boundaries by Reward")
    ax.legend(loc="lower right")

    # Plot 2: Smoothed success rate with stage colors
    ax = axes[1]
    ax.plot(ep_sorted["training_step"], ep_sorted["smoothed_success"], color="black", linewidth=1)
    for stage_idx, stage_name in enumerate(labels):
        stage_eps = ep_sorted[ep_sorted["stage_idx"] == stage_idx]
        ax.scatter(stage_eps["training_step"], stage_eps["smoothed_success"],
                   alpha=0.3, s=5, color=colors[stage_idx])

    for b in stage_boundaries[1:-1]:
        ax.axvline(b, color="black", linestyle="--", alpha=0.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Smoothed Success Rate")
    ax.set_title("Stage Boundaries by Success Rate")
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[5/5] Saved stage visualization: {output_path}")


def main():
    args = parse_args()

    # Load data
    obs, act, slots = load_and_align_data(
        args.obs_csv, args.act_csv, args.slot_pkl, args.video_key
    )

    # Detect episodes
    print(f"[4/5] Detecting episodes (min_length={args.min_episode_length})")
    ep_df = detect_episodes(obs, min_length=args.min_episode_length)
    print(f"      Found {len(ep_df)} valid episodes")

    # Choose success definition
    if args.success_metric == "reward":
        ep_df["success"] = ep_df["total_reward"] > 0
    elif args.success_metric == "improvement":
        ep_df["success"] = ep_df["improvement"] > 0
    elif args.success_metric == "final_dist":
        ep_df["success"] = ep_df["end_goal_dist"] <= args.dist_thresh
    else:
        ep_df["success"] = ep_df["total_reward"] > 0
    print(f"      Using success_metric={args.success_metric}")

    # Assign stages
    if args.method == "adaptive":
        ep_df, boundaries, labels = assign_stages_adaptive(
            ep_df, args.n_stages, args.smooth_window
        )
    elif args.method == "percentile":
        ep_df, boundaries, labels = assign_stages_percentile(ep_df, args.n_stages)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"      Stage boundaries (training_step): {boundaries}")
    for i, label in enumerate(labels):
        stage_eps = ep_df[ep_df["stage_idx"] == i]
        if len(stage_eps) > 0:
            print(f"      {label}: {len(stage_eps)} episodes, "
                  f"success={stage_eps['success'].mean()*100:.1f}%, "
                  f"reward={stage_eps['total_reward'].mean():.3f}")

    # Write pickles
    print(f"[5/5] Writing stage pickles to {args.output_dir}")
    written = write_stage_pickles(
        ep_df, obs, act, slots, args.output_dir, args.video_key, labels
    )

    # Visualize
    if args.visualize:
        viz_path = Path(args.output_dir) / f"{args.video_key}_stages.png"
        visualize_stages(ep_df, boundaries, labels, str(viz_path))

    # Summary
    print("\n=== Summary ===")
    print(f"Output directory: {args.output_dir}")
    for stage_name, base_path in written:
        print(f"  {stage_name}:")
        print(f"    slots:  {base_path}_slots.pkl")
        print(f"    actions: {base_path}_actions.pkl")
        print(f"    proprio: {base_path}_proprio.pkl")
        print(f"    state:   {base_path}_state.pkl")
        print(f"    reward:  {base_path}_reward.pkl")
        print(f"    meta:    {base_path}_meta.json")

    print("\nNext step: run validation on each stage:")
    print(f"  python scripts/validate_mlagents_counterfactual.py \\")
    print(f"      --config configs/config_train_causal_pusht_slot.yaml \\")
    print(f"      --checkpoint <your_ckpt> \\")
    print(f"      --embedding_dir {args.output_dir}/{args.video_key}_stage_X_<name>_slots.pkl \\")
    print(f"      --action_dir {args.output_dir}/{args.video_key}_stage_X_<name>_actions.pkl \\")
    print(f"      --proprio_dir {args.output_dir}/{args.video_key}_stage_X_<name>_proprio.pkl")


if __name__ == "__main__":
    main()
