import argparse
import json
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs", required=True, help="Observations CSV path")
    parser.add_argument("--out", default="checkpoints/episodes_with_reward_train03.json")
    args = parser.parse_args()

    df = pd.read_csv(args.obs)
    # remove init rows if present
    if "agent_id" in df.columns:
        df = df[df["agent_id"] != "init"].reset_index(drop=True)

    grouped = df.groupby("episode_id")
    episodes = []
    for ep_id, g in grouped:
        total_reward = float(g["reward"].sum())
        if total_reward > 0:
            episodes.append({
                "episode_id": int(ep_id),
                "total_reward": total_reward,
                "n_frames": int(len(g)),
                "start_idx": int(g.index[0]),
                "end_idx": int(g.index[-1]),
                "training_step_min": int(g["training_step"].min()) if "training_step" in g.columns else None,
                "training_step_max": int(g["training_step"].max()) if "training_step" in g.columns else None,
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n_episodes_with_reward": len(episodes), "episodes": episodes}, f, indent=2)

    print(f"Found {len(episodes)} episodes with total_reward > 0")
    for e in episodes:
        print(f"ep {e['episode_id']}: reward={e['total_reward']:.3f}, frames={e['n_frames']}, range={e['start_idx']}-{e['end_idx']}, steps={e['training_step_min']}-{e['training_step_max']}")


if __name__ == '__main__':
    main()
