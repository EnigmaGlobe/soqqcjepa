import csv
import json
from pathlib import Path

csv_path = Path('checkpoints/val_by_stage.csv')
out_txt = Path('checkpoints/val_by_stage_summary.txt')
out_json = Path('checkpoints/val_by_stage_summary.json')

rows = []
with csv_path.open() as fh:
    r = csv.DictReader(fh)
    for row in r:
        # parse numeric fields
        parsed = {'stage': row['stage']}
        for k,v in row.items():
            if k == 'stage':
                continue
            if v is None or v == '':
                parsed[k] = None
                continue
            # parse list-like fields
            if k in ('per_slot_action_sensitivity','slot_ablation_importance'):
                s = v.strip('"')
                parsed[k] = [float(x) for x in s.split(';') if x!='']
            else:
                try:
                    parsed[k] = float(v)
                except Exception:
                    parsed[k] = v
        rows.append(parsed)

# compute summary
summary = {'stages':{}, 'best_stage_by':{}}
for r in rows:
    s = r['stage']
    entry = {
        'val_future_mse': r.get('val_future_mse'),
        'val_future_std': r.get('val_future_std'),
        'num_pairs': int(r.get('num_pairs')) if r.get('num_pairs') is not None else None,
        'action_zero_divergence_mse': r.get('action_zero_divergence_mse'),
        'action_random_divergence_mse': r.get('action_random_divergence_mse'),
        'action_history_swap_divergence_mse': r.get('action_history_swap_divergence_mse'),
        'per_slot_action_sensitivity_mean': None,
        'slot_ablation_importance_mean': None,
        'most_causal_slot_index': r.get('most_causal_slot_index'),
        'least_causal_slot_index': r.get('least_causal_slot_index'),
    }
    pas = r.get('per_slot_action_sensitivity')
    sai = r.get('slot_ablation_importance')
    if isinstance(pas, list) and pas:
        entry['per_slot_action_sensitivity_mean'] = sum(pas)/len(pas)
    if isinstance(sai, list) and sai:
        entry['slot_ablation_importance_mean'] = sum(sai)/len(sai)
    summary['stages'][s] = entry

# identify best/worst by val_future_mse
valid = [(s,e['val_future_mse']) for s,e in summary['stages'].items() if e['val_future_mse'] is not None]
if valid:
    best = min(valid, key=lambda x: x[1])
    worst = max(valid, key=lambda x: x[1])
    summary['best_stage_by']['val_future_mse'] = {'stage': best[0], 'val_future_mse': best[1]}
    summary['worst_stage_by'] = {'stage': worst[0], 'val_future_mse': worst[1]}

# write outputs
with out_txt.open('w') as fh:
    fh.write('Validation by stage summary\n')
    fh.write('========================\n\n')
    for s,e in summary['stages'].items():
        fh.write(f"Stage: {s}\n")
        fh.write(f"  val_future_mse: {e['val_future_mse']} (std {e['val_future_std']})\n")
        fh.write(f"  num_pairs: {e['num_pairs']}\n")
        fh.write(f"  action_zero_divergence_mse: {e['action_zero_divergence_mse']}\n")
        fh.write(f"  action_random_divergence_mse: {e['action_random_divergence_mse']}\n")
        fh.write(f"  per_slot_action_sensitivity_mean: {e['per_slot_action_sensitivity_mean']}\n")
        fh.write(f"  slot_ablation_importance_mean: {e['slot_ablation_importance_mean']}\n")
        fh.write(f"  most_causal_slot_index: {e['most_causal_slot_index']}, least_causal_slot_index: {e['least_causal_slot_index']}\n")
        fh.write('\n')
    if 'best_stage_by' in summary and summary['best_stage_by']:
        fh.write('Best stage by val_future_mse:\n')
        fh.write(f"  {summary['best_stage_by']['val_future_mse']['stage']} ({summary['best_stage_by']['val_future_mse']['val_future_mse']})\n")

with out_json.open('w') as fh:
    json.dump(summary, fh, indent=2)

print('Wrote', out_txt, 'and', out_json)
