import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def compute_per_episode(df):
    # assumes df has agent_pos_x/z and goal_pos_x/z and episode_id, reward, training_step
    eps = []
    for ep_id, g in df.groupby('episode_id'):
        start = g.index[0]
        end = g.index[-1]
        agent_end = np.array([g['agent_pos_x'].iloc[-1], g['agent_pos_z'].iloc[-1]])
        goal = np.array([g['goal_pos_x'].iloc[0], g['goal_pos_z'].iloc[0]])
        end_goal_dist = float(np.linalg.norm(agent_end - goal))
        improvement = float((np.linalg.norm(np.array([g['agent_pos_x'].iloc[0], g['agent_pos_z'].iloc[0]]) - goal) - end_goal_dist))
        total_reward = float(g['reward'].sum())
        max_reward = float(g['reward'].max())
        eps.append({
            'episode_id': int(ep_id),
            'start_idx': int(start),
            'end_idx': int(end),
            'n_frames': int(len(g)),
            'training_step_min': int(g['training_step'].min()) if 'training_step' in g.columns else None,
            'training_step_max': int(g['training_step'].max()) if 'training_step' in g.columns else None,
            'total_reward': total_reward,
            'max_reward': max_reward,
            'end_goal_dist': end_goal_dist,
            'improvement': improvement,
        })
    return eps


def load_stage_meta(stage_meta_dir, video_key):
    p = Path(stage_meta_dir)
    metas = {}
    for f in p.glob(f"{video_key}*meta.json"):
        try:
            j = json.load(open(f, 'r'))
            metas[j['stage_idx']] = j
        except Exception:
            continue
    return metas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs', required=True)
    parser.add_argument('--stage_meta_dir', default='checkpoints/validation_staged')
    parser.add_argument('--video_key', default='train_03_recording')
    parser.add_argument('--dist_thresh', type=float, default=1.0, help='Distance threshold for success')
    parser.add_argument('--out', default='checkpoints/episode_success_metrics.json')
    args = parser.parse_args()

    df = pd.read_csv(args.obs)
    if 'agent_id' in df.columns:
        df = df[df['agent_id'] != 'init'].reset_index(drop=True)

    per_ep = compute_per_episode(df)

    # compute success flags
    for e in per_ep:
        e['success_by_reward_gt0'] = e['total_reward'] > 0
        e['success_by_final_dist'] = e['end_goal_dist'] <= args.dist_thresh
        e['success_by_improvement'] = e['improvement'] > 0

    # load stage metas and map eps to stages
    metas = load_stage_meta(args.stage_meta_dir, args.video_key)
    ep_stage = {}
    for si, m in metas.items():
        for ep in m.get('episode_ids', []):
            ep_stage[int(ep)] = si

    # aggregate per-stage
    stages = {}
    for e in per_ep:
        sid = ep_stage.get(e['episode_id'], None)
        stages.setdefault(sid, []).append(e)

    summary = {'total_episodes': len(per_ep), 'dist_thresh': args.dist_thresh, 'stages': {}}
    for sid, eps in stages.items():
        n = len(eps)
        r1 = sum(1 for x in eps if x['success_by_reward_gt0']) / n
        r2 = sum(1 for x in eps if x['success_by_final_dist']) / n
        r3 = sum(1 for x in eps if x['success_by_improvement']) / n
        summary['stages'][str(sid)] = {
            'n_episodes': n,
            'success_reward_gt0': r1,
            'success_final_dist': r2,
            'success_improvement': r3,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'per_episode': per_ep, 'summary': summary}, f, indent=2)

    print('Wrote', args.out)
    print('Summary:')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
