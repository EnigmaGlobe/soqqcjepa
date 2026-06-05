#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from semopy import Model, calc_stats


EPS = 1e-8


def zscore(series):
    series = pd.Series(series, copy=False)
    std = float(series.std(ddof=0))
    if std <= 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - float(series.mean())) / std


def iqr(series):
    values = pd.Series(series).dropna().astype(float)
    if values.empty:
        return np.nan
    return float(values.quantile(0.75) - values.quantile(0.25))


def cohens_d(a, b):
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    pooled_num = (len(a) - 1) * var_a + (len(b) - 1) * var_b
    pooled_den = len(a) + len(b) - 2
    if pooled_den <= 0:
        return np.nan
    pooled_std = np.sqrt(max(pooled_num / pooled_den, 0.0))
    if pooled_std <= 0:
        return np.nan
    return float((a.mean() - b.mean()) / pooled_std)


def cronbach_alpha(frame):
    data = frame.dropna().astype(float)
    if data.shape[0] < 2 or data.shape[1] < 2:
        return np.nan
    item_vars = data.var(axis=0, ddof=1)
    total = data.sum(axis=1)
    total_var = float(total.var(ddof=1))
    if total_var <= 0 or np.isnan(total_var):
        return np.nan
    k = data.shape[1]
    return float((k / (k - 1.0)) * (1.0 - float(item_vars.sum()) / total_var))


def compute_cfi(frame):
    data = frame.dropna().astype(float)
    if data.shape[0] < 20:
        return np.nan
    data = data.loc[:, data.std(axis=0, ddof=0) > 0]
    if data.shape[1] < 3:
        return np.nan

    model_desc = "dataset_quality =~ " + " + ".join(list(data.columns))
    model = Model(model_desc)
    try:
        model.fit(data)
        stats = calc_stats(model)
    except Exception:
        return np.nan

    if isinstance(stats, pd.DataFrame):
        if 'CFI' in stats.columns:
            value = stats['CFI'].iloc[0]
            return float(value) if pd.notna(value) else np.nan
        if 'CFI' in stats.index and 'Value' in stats.columns:
            value = stats.loc['CFI', 'Value']
            return float(value) if pd.notna(value) else np.nan
    if isinstance(stats, dict) and 'CFI' in stats:
        value = stats['CFI']
        return float(value) if value is not None else np.nan
    return np.nan


def load_episode_level_metrics(observations_csv):
    df = pd.read_csv(observations_csv)
    df = df[df['episode_id'] > 0].copy()
    if 'agent_id' in df.columns:
        df = df[df['agent_id'] != 'init'].copy()

    sort_cols = [col for col in ['episode_id', 'frame_count', 'training_step', 'step_index'] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    df['goal_distance'] = np.sqrt(
        (df['block_pos_x'] - df['goal_pos_x']) ** 2
        + (df['block_pos_y'] - df['goal_pos_y']) ** 2
        + (df['block_pos_z'] - df['goal_pos_z']) ** 2
    )

    rows = []
    for episode_id, group in df.groupby('episode_id', sort=True):
        group = group.sort_values([col for col in ['frame_count', 'training_step', 'step_index'] if col in group.columns])
        valid_goal_distance = group['goal_distance'].dropna()
        if valid_goal_distance.empty:
            continue
        start_goal_distance = float(valid_goal_distance.iloc[0])
        best_goal_distance = float(valid_goal_distance.min())
        final_goal_distance = float(valid_goal_distance.iloc[-1])
        episode_steps = int(group['step_index'].max()) if 'step_index' in group.columns else int(len(group) - 1)
        max_reward = float(group['reward'].max())
        reward_sum = float(group['reward'].sum())
        success_proxy = float(max_reward >= 4.9)
        normalized_task_progress = float((start_goal_distance - best_goal_distance) / max(start_goal_distance, EPS))
        progress_rate = float(normalized_task_progress / max(episode_steps, 1))
        rows.append({
            'episode_id': int(episode_id),
            'start_goal_distance': start_goal_distance,
            'best_goal_distance': best_goal_distance,
            'final_goal_distance': final_goal_distance,
            'episode_steps': episode_steps,
            'max_reward': max_reward,
            'reward_sum': reward_sum,
            'success_proxy': success_proxy,
            'normalized_task_progress': normalized_task_progress,
            'progress_rate': progress_rate,
        })

    episode_df = pd.DataFrame(rows).sort_values('episode_id').reset_index(drop=True)
    episode_df['time_score'] = zscore(episode_df['episode_id'])
    episode_df['reward_score'] = zscore(episode_df['max_reward'])
    episode_df['distance_score'] = zscore(-episode_df['best_goal_distance'])
    return episode_df


def exact_half_split(sorted_df):
    sorted_df = sorted_df.reset_index(drop=True)
    omitted_episode_id = None
    if len(sorted_df) % 2 == 1:
        mid = len(sorted_df) // 2
        omitted_episode_id = int(sorted_df.iloc[mid]['episode_id'])
        sorted_df = pd.concat([sorted_df.iloc[:mid], sorted_df.iloc[mid + 1:]], ignore_index=True)
    half = len(sorted_df) // 2
    group_a = sorted_df.iloc[:half].copy()
    group_b = sorted_df.iloc[half:].copy()
    return group_a, group_b, omitted_episode_id


def rank_split(episode_df, score_series, ascending, phase_name, iv_name, group_a_name, group_b_name):
    ranked = episode_df.assign(_score=score_series).sort_values(['_score', 'episode_id'], ascending=[ascending, True]).reset_index(drop=True)
    group_a, group_b, omitted_episode_id = exact_half_split(ranked)
    return {
        'phase': phase_name,
        'iv': iv_name,
        'group_a_name': group_a_name,
        'group_b_name': group_b_name,
        'group_a': group_a.drop(columns=['_score']),
        'group_b': group_b.drop(columns=['_score']),
        'omitted_episode_id': omitted_episode_id,
    }


def random_splits(episode_df, repeats, seed):
    splits = []
    for repeat_idx in range(repeats):
        shuffled = episode_df.sample(frac=1.0, random_state=seed + repeat_idx).reset_index(drop=True)
        group_a, group_b, omitted_episode_id = exact_half_split(shuffled)
        splits.append({
            'phase': 'phase5_random',
            'iv': 'random',
            'repeat': repeat_idx + 1,
            'group_a_name': 'random_set_a',
            'group_b_name': 'random_set_b',
            'group_a': group_a,
            'group_b': group_b,
            'omitted_episode_id': omitted_episode_id,
        })
    return splits


def indicator_frame(group_df):
    items = pd.DataFrame({
        'success_proxy': group_df['success_proxy'].astype(float),
        'normalized_task_progress': group_df['normalized_task_progress'].astype(float),
        'neg_final_goal_distance': -group_df['final_goal_distance'].astype(float),
        'progress_rate': group_df['progress_rate'].astype(float),
        'reward_sum': group_df['reward_sum'].astype(float),
    })
    return items.apply(zscore)


def summarize_group(group_df, phase, iv_name, group_name, repeat=None):
    items = indicator_frame(group_df)
    summary = {
        'phase': phase,
        'iv': iv_name,
        'group_name': group_name,
        'repeat': repeat,
        'n_episodes': int(len(group_df)),
        'success_proxy_mean': float(group_df['success_proxy'].mean()),
        'success_proxy_sd': float(group_df['success_proxy'].std(ddof=1)),
        'normalized_task_progress_mean': float(group_df['normalized_task_progress'].mean()),
        'normalized_task_progress_sd': float(group_df['normalized_task_progress'].std(ddof=1)),
        'final_goal_distance_mean': float(group_df['final_goal_distance'].mean()),
        'final_goal_distance_sd': float(group_df['final_goal_distance'].std(ddof=1)),
        'progress_rate_mean': float(group_df['progress_rate'].mean()),
        'progress_rate_sd': float(group_df['progress_rate'].std(ddof=1)),
        'reward_sum_mean': float(group_df['reward_sum'].mean()),
        'reward_sum_sd': float(group_df['reward_sum'].std(ddof=1)),
        'reward_sum_iqr': iqr(group_df['reward_sum']),
        'alpha': cronbach_alpha(items),
        'cfi': compute_cfi(items),
    }
    return summary


def compare_groups(split_info):
    repeat = split_info.get('repeat')
    summary_a = summarize_group(split_info['group_a'], split_info['phase'], split_info['iv'], split_info['group_a_name'], repeat)
    summary_b = summarize_group(split_info['group_b'], split_info['phase'], split_info['iv'], split_info['group_b_name'], repeat)

    compare_row = {
        'phase': split_info['phase'],
        'iv': split_info['iv'],
        'repeat': repeat,
        'group_a_name': split_info['group_a_name'],
        'group_b_name': split_info['group_b_name'],
        'omitted_episode_id': split_info.get('omitted_episode_id'),
        'n_a': summary_a['n_episodes'],
        'n_b': summary_b['n_episodes'],
    }

    metrics = [
        'success_proxy_mean',
        'normalized_task_progress_mean',
        'final_goal_distance_mean',
        'progress_rate_mean',
        'reward_sum_sd',
        'reward_sum_iqr',
        'alpha',
        'cfi',
    ]
    for metric in metrics:
        compare_row[f'a_{metric}'] = summary_a[metric]
        compare_row[f'b_{metric}'] = summary_b[metric]
        compare_row[f'diff_{metric}'] = summary_a[metric] - summary_b[metric]

    effect_metrics = [
        ('success_proxy', 'success_proxy_effect_d'),
        ('normalized_task_progress', 'normalized_task_progress_effect_d'),
        ('final_goal_distance', 'final_goal_distance_effect_d'),
        ('progress_rate', 'progress_rate_effect_d'),
        ('reward_sum', 'reward_sum_effect_d'),
    ]
    for column, effect_name in effect_metrics:
        compare_row[effect_name] = cohens_d(split_info['group_a'][column], split_info['group_b'][column])

    return summary_a, summary_b, compare_row


def add_quality_scores(group_metrics_df):
    scored = group_metrics_df.copy()
    scored['quality_score'] = (
        zscore(scored['success_proxy_mean'])
        + zscore(scored['normalized_task_progress_mean'])
        + zscore(-scored['final_goal_distance_mean'])
        + zscore(scored['progress_rate_mean'])
        + zscore(-scored['reward_sum_sd'])
        + zscore(scored['alpha'].fillna(scored['alpha'].mean()))
        + zscore(scored['cfi'].fillna(scored['cfi'].mean()))
    ) / 7.0
    return scored


def add_pair_scores(comparison_df):
    scored = comparison_df.copy()
    scored['pair_score'] = (
        zscore(scored['diff_success_proxy_mean'])
        + zscore(scored['diff_normalized_task_progress_mean'])
        + zscore(-scored['diff_final_goal_distance_mean'])
        + zscore(scored['diff_progress_rate_mean'])
        + zscore(-scored['diff_reward_sum_sd'])
        + zscore(-scored['diff_reward_sum_iqr'])
        + zscore(((scored['a_alpha'] + scored['b_alpha']) / 2.0).fillna(((scored['a_alpha'] + scored['b_alpha']) / 2.0).mean()))
        + zscore(((scored['a_cfi'] + scored['b_cfi']) / 2.0).fillna(((scored['a_cfi'] + scored['b_cfi']) / 2.0).mean()))
    ) / 8.0
    return scored


def build_pair_recommendation(pair_df, output_path):
    eligible = pair_df[pair_df['phase'] != 'phase5_random'].copy()
    eligible = eligible.sort_values('pair_score', ascending=False).reset_index(drop=True)
    best = eligible.iloc[0]

    lines = []
    lines.append('# Final Phase Pair Recommendation')
    lines.append('')
    lines.append('Recommended matched phase combination')
    lines.append(f"- Phase: {best['phase']}")
    lines.append(f"- IV: {best['iv']}")
    lines.append(f"- Group A: {best['group_a_name']}")
    lines.append(f"- Group B: {best['group_b_name']}")
    lines.append(f"- Episodes per group: {int(best['n_a'])}")
    lines.append('')
    lines.append('Why this pair is recommended')
    lines.append(f"- Success proxy separation (A-B): {best['diff_success_proxy_mean']:.4f}")
    lines.append(f"- Progress separation (A-B): {best['diff_normalized_task_progress_mean']:.4f}")
    lines.append(f"- Final goal distance separation (A-B): {best['diff_final_goal_distance_mean']:.4f}")
    lines.append(f"- Progress rate separation (A-B): {best['diff_progress_rate_mean']:.8f}")
    lines.append(f"- Reward SD separation (A-B): {best['diff_reward_sum_sd']:.4f}")
    avg_alpha = (best['a_alpha'] + best['b_alpha']) / 2.0
    avg_cfi = (best['a_cfi'] + best['b_cfi']) / 2.0
    lines.append(f"- Average alpha across pair: {avg_alpha:.4f}" if pd.notna(avg_alpha) else '- Average alpha across pair: NaN')
    lines.append(f"- Average CFI across pair: {avg_cfi:.4f}" if pd.notna(avg_cfi) else '- Average CFI across pair: NaN')
    lines.append(f"- Pair score: {best['pair_score']:.4f}")
    lines.append('')
    lines.append('Interpretation')
    lines.append('- This recommendation treats each phase as a matched A/B combination rather than choosing a single group in isolation.')
    lines.append('- The best pair is the split with the strongest between-group separation in the expected direction while keeping the two groups reasonably coherent.')
    lines.append('')
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def build_final_recommendation(group_metrics_df, comparison_df, output_path):
    eligible = group_metrics_df[group_metrics_df['phase'] != 'phase5_random'].copy()
    eligible = eligible.sort_values('quality_score', ascending=False).reset_index(drop=True)
    best = eligible.iloc[0]

    lines = []
    lines.append('# Final Dataset Recommendation')
    lines.append('')
    lines.append('Recommended candidate subset')
    lines.append(f"- Phase: {best['phase']}")
    lines.append(f"- IV: {best['iv']}")
    lines.append(f"- Group: {best['group_name']}")
    lines.append(f"- Episodes: {int(best['n_episodes'])}")
    lines.append('')
    lines.append('Why this group is recommended')
    lines.append(f"- Success proxy mean: {best['success_proxy_mean']:.4f}")
    lines.append(f"- Normalized task progress mean: {best['normalized_task_progress_mean']:.4f}")
    lines.append(f"- Final goal distance mean: {best['final_goal_distance_mean']:.4f}")
    lines.append(f"- Progress rate mean: {best['progress_rate_mean']:.8f}")
    lines.append(f"- Reward SD: {best['reward_sum_sd']:.4f}")
    lines.append(f"- Cronbach alpha: {best['alpha']:.4f}" if pd.notna(best['alpha']) else '- Cronbach alpha: NaN')
    lines.append(f"- CFI: {best['cfi']:.4f}" if pd.notna(best['cfi']) else '- CFI: NaN')
    lines.append(f"- Composite quality score: {best['quality_score']:.4f}")
    lines.append('')
    lines.append('Interpretation')
    lines.append('- This recommendation follows the FRD scoring framework rather than a manual early-vs-late selection.')
    lines.append('- The selected subset is the strongest combined candidate under progress, goal proximity, efficiency, and reliability-style checks.')
    lines.append('')
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observations_csv', required=True)
    parser.add_argument('--output_dir', default='outputs/dataset_selection')
    parser.add_argument('--random_repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_df = load_episode_level_metrics(args.observations_csv)
    episode_df.to_csv(output_dir / 'episode_level_metrics.csv', index=False)

    splits = []
    splits.append(rank_split(episode_df, episode_df['episode_id'], True, 'phase1_time', 'time', 'first_half', 'second_half'))
    splits.append(rank_split(episode_df, episode_df['max_reward'], False, 'phase2_reward', 'reward', 'top_reward', 'bottom_reward'))
    splits.append(rank_split(episode_df, episode_df['best_goal_distance'], True, 'phase3_distance', 'distance', 'closer_to_goal', 'farther_from_goal'))

    combined_tr = (episode_df['time_score'] + episode_df['reward_score']) / 2.0
    combined_td = (episode_df['time_score'] + episode_df['distance_score']) / 2.0
    combined_rd = (episode_df['reward_score'] + episode_df['distance_score']) / 2.0
    combined_trd = (episode_df['time_score'] + episode_df['reward_score'] + episode_df['distance_score']) / 3.0
    splits.append(rank_split(episode_df, combined_tr, False, 'phase4a_time_reward', 'time+reward', 'top_combined', 'bottom_combined'))
    splits.append(rank_split(episode_df, combined_td, False, 'phase4b_time_distance', 'time+distance', 'top_combined', 'bottom_combined'))
    splits.append(rank_split(episode_df, combined_rd, False, 'phase4c_reward_distance', 'reward+distance', 'top_combined', 'bottom_combined'))
    splits.append(rank_split(episode_df, combined_trd, False, 'phase4d_time_reward_distance', 'time+reward+distance', 'top_combined', 'bottom_combined'))
    splits.extend(random_splits(episode_df, args.random_repeats, args.seed))

    group_summaries = []
    comparison_rows = []
    for split in splits:
        summary_a, summary_b, compare_row = compare_groups(split)
        group_summaries.extend([summary_a, summary_b])
        comparison_rows.append(compare_row)

    group_metrics_df = pd.DataFrame(group_summaries)
    group_metrics_df = add_quality_scores(group_metrics_df)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = add_pair_scores(comparison_df)

    reliability_cols = ['phase', 'iv', 'group_name', 'repeat', 'n_episodes', 'alpha', 'cfi', 'quality_score']
    group_metrics_df.to_csv(output_dir / 'candidate_group_metrics.csv', index=False)
    comparison_df.to_csv(output_dir / 'phase_comparison_summary.csv', index=False)
    group_metrics_df[reliability_cols].to_csv(output_dir / 'reliability_fit_summary.csv', index=False)
    comparison_df.to_csv(output_dir / 'phase_pair_summary.csv', index=False)

    build_final_recommendation(group_metrics_df, comparison_df, output_dir / 'final_dataset_recommendation.md')
    build_pair_recommendation(comparison_df, output_dir / 'final_phase_pair_recommendation.md')

    metadata = {
        'observations_csv': str(args.observations_csv),
        'output_dir': str(output_dir),
        'random_repeats': args.random_repeats,
        'seed': args.seed,
        'n_episodes': int(len(episode_df)),
        'deterministic_phases': 7,
    }
    (output_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(f'Wrote {output_dir / "episode_level_metrics.csv"}')
    print(f'Wrote {output_dir / "phase_comparison_summary.csv"}')
    print(f'Wrote {output_dir / "phase_pair_summary.csv"}')
    print(f'Wrote {output_dir / "reliability_fit_summary.csv"}')
    print(f'Wrote {output_dir / "final_dataset_recommendation.md"}')
    print(f'Wrote {output_dir / "final_phase_pair_recommendation.md"}')


if __name__ == '__main__':
    main()