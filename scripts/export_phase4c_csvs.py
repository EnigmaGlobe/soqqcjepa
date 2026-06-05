from pathlib import Path
import numpy as np
import pandas as pd

from analyze_dataset_selection import load_episode_level_metrics, rank_split


def add_reliability_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched['neg_final_goal_distance'] = -enriched['final_goal_distance'].astype(float)
    enriched['neg_best_goal_distance'] = -enriched['best_goal_distance'].astype(float)
    enriched['log_progress_rate'] = np.log(enriched['progress_rate'].astype(float).clip(lower=1e-12))
    enriched['log_reward_gap'] = -np.log((5.000001 - enriched['reward_sum'].astype(float)).clip(lower=1e-12))
    return enriched


def main():
    observations_csv = Path('testdata/1/data_training/observations_frame_train_01 - Copy.csv')
    output_dir = Path('outputs/dataset_selection')
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_df = load_episode_level_metrics(observations_csv)
    combined_rd = (episode_df['reward_score'] + episode_df['distance_score']) / 2.0
    split = rank_split(
        episode_df,
        combined_rd,
        False,
        'phase4c_reward_distance',
        'reward+distance',
        'top_combined',
        'bottom_combined',
    )

    top_path = output_dir / 'phase4c_reward_distance_top_combined.csv'
    bottom_path = output_dir / 'phase4c_reward_distance_bottom_combined.csv'
    top_group = add_reliability_columns(split['group_a'])
    bottom_group = add_reliability_columns(split['group_b'])

    top_group.to_csv(top_path, index=False)
    bottom_group.to_csv(bottom_path, index=False)

    summary_path = output_dir / 'phase4c_reward_distance_split_summary.txt'
    summary_path.write_text(
        '\n'.join([
            f"omitted_episode_id={split['omitted_episode_id']}",
            f"top_count={len(top_group)}",
            f"bottom_count={len(bottom_group)}",
            f"top_csv={top_path}",
            f"bottom_csv={bottom_path}",
            "added_columns=neg_final_goal_distance,neg_best_goal_distance,log_progress_rate,log_reward_gap",
        ]),
        encoding='utf-8',
    )

    print(f'Wrote {top_path}')
    print(f'Wrote {bottom_path}')
    print(f'Wrote {summary_path}')


if __name__ == '__main__':
    main()
