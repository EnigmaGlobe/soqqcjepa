from pathlib import Path

import numpy as np
import pandas as pd


def build_episode_metrics(observations_csv: Path) -> pd.DataFrame:
    observations = pd.read_csv(observations_csv)
    observations = observations[
        (observations['episode_id'] > 0) & (observations['agent_id'] != 'init')
    ].copy()

    observations['goal_distance'] = np.sqrt(
        (observations['block_pos_x'] - observations['goal_pos_x']) ** 2
        + (observations['block_pos_y'] - observations['goal_pos_y']) ** 2
        + (observations['block_pos_z'] - observations['goal_pos_z']) ** 2
    )

    rows = []
    for episode_id, group in observations.groupby('episode_id', sort=True):
        group = group.sort_values('step_index').reset_index(drop=True)
        valid_distances = group['goal_distance'].dropna()
        if valid_distances.empty:
            continue

        start_distance = float(valid_distances.iloc[0])
        best_distance = float(valid_distances.min())
        final_distance = float(valid_distances.iloc[-1])

        if start_distance > 0:
            final_progress_ratio = (start_distance - final_distance) / start_distance
        else:
            final_progress_ratio = np.nan

        rows.append(
            {
                'episode_id': int(episode_id),
                'neg_final_goal_distance': -final_distance,
                'neg_best_goal_distance': -best_distance,
                'final_progress_ratio': final_progress_ratio,
                'neg_goal_regression': -(final_distance - best_distance),
            }
        )

    return pd.DataFrame(rows).sort_values('episode_id').reset_index(drop=True)


def main() -> None:
    data_dir = Path('testdata/1/data_training')
    observations_csv = data_dir / 'observations_frame_train_01.csv'
    output_csv = data_dir / 'comboA_performance_retention_episode_metrics.csv'

    metrics = build_episode_metrics(observations_csv)
    metrics.to_csv(output_csv, index=False)

    print(f'Wrote {output_csv}')
    print(f'Rows: {len(metrics)}')
    print('Columns: episode_id, neg_final_goal_distance, neg_best_goal_distance, final_progress_ratio, neg_goal_regression')


if __name__ == '__main__':
    main()