#!/usr/bin/env python
"""Classify episodes by observed behavior and optionally regroup folders.

Labels:
- well_trained: max_reward >= 4.9
- exploring: max_reward < 1.0 and goal_progress < 1.0
- transition: everything else

This is episode-by-episode classification, not training-range classification.
"""
from pathlib import Path
import argparse
import shutil

import numpy as np
import pandas as pd


def load_episode_metrics(observations_csv: Path) -> pd.DataFrame:
    obs = pd.read_csv(observations_csv)
    obs = obs[obs['episode_id'] > 0].copy()
    obs['goal_dist'] = np.sqrt(
        (obs['block_pos_x'] - obs['goal_pos_x']) ** 2
        + (obs['block_pos_y'] - obs['goal_pos_y']) ** 2
        + (obs['block_pos_z'] - obs['goal_pos_z']) ** 2
    )

    by_ep = obs.groupby('episode_id').agg(
        max_reward=('reward', 'max'),
        reward_sum=('reward', 'sum'),
        start_goal_dist=('goal_dist', 'first'),
        min_goal_dist=('goal_dist', 'min'),
        final_goal_dist=('goal_dist', 'last'),
        steps=('step_index', 'max'),
    )
    by_ep['goal_progress'] = by_ep['start_goal_dist'] - by_ep['min_goal_dist']
    by_ep['label'] = 'transition'
    by_ep.loc[by_ep['max_reward'] >= 4.9, 'label'] = 'well_trained'
    by_ep.loc[(by_ep['max_reward'] < 1.0) & (by_ep['goal_progress'] < 1.0), 'label'] = 'exploring'
    return by_ep.reset_index()


def move_matching_episode_dirs(source_dirs, dest_root: Path, labels_df: pd.DataFrame):
    label_lookup = dict(zip(labels_df['episode_id'].astype(int), labels_df['label']))
    moved = {'exploring': 0, 'well_trained': 0, 'transition': 0}

    for source_dir in source_dirs:
        source_dir = Path(source_dir)
        if not source_dir.exists():
            continue

        for episode_dir in sorted(source_dir.glob('episode_*')):
            try:
                episode_num = int(episode_dir.name.split('_', 1)[1])
            except Exception:
                continue

            label = label_lookup.get(episode_num)
            if label not in ('exploring', 'well_trained'):
                continue

            target_dir = dest_root / label / episode_dir.name
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            if target_dir.exists():
                continue
            shutil.move(str(episode_dir), str(target_dir))
            moved[label] += 1

    return moved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observations_csv', required=True)
    parser.add_argument('--report_csv', required=True)
    parser.add_argument('--dest_root', required=True)
    parser.add_argument('--source_dirs', nargs='*', default=[])
    args = parser.parse_args()

    metrics = load_episode_metrics(Path(args.observations_csv))
    report_csv = Path(args.report_csv)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report_csv, index=False)

    moved = {'exploring': 0, 'well_trained': 0, 'transition': 0}
    if args.source_dirs:
        moved = move_matching_episode_dirs(args.source_dirs, Path(args.dest_root), metrics)

    print('Wrote report to', report_csv)
    print(metrics['label'].value_counts().sort_index().to_string())
    print('Moved exploring:', moved['exploring'])
    print('Moved well_trained:', moved['well_trained'])


if __name__ == '__main__':
    main()
