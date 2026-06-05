import csv
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

CSV_PATH = 'checkpoints/val_by_stage.csv'
OUT_DIR = 'outputs/val_by_stage_tb'

def parse_list_field(s):
    s = s.strip().strip('"')
    if s == '':
        return []
    return [float(x) for x in s.split(';')]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    run_dir = os.path.join(OUT_DIR, datetime.now().strftime('run_%Y%m%d_%H%M%S'))
    writer = SummaryWriter(run_dir)

    with open(CSV_PATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            stage = row['stage']
            tag_prefix = f"stage/{stage}"
            # Scalars
            writer.add_scalar(f"{tag_prefix}/val_future_mse", float(row['val_future_mse']), i)
            writer.add_scalar(f"{tag_prefix}/val_future_std", float(row['val_future_std']), i)
            writer.add_scalar(f"{tag_prefix}/action_zero_divergence_mse", float(row['action_zero_divergence_mse']), i)
            writer.add_scalar(f"{tag_prefix}/action_random_divergence_mse", float(row['action_random_divergence_mse']), i)
            writer.add_scalar(f"{tag_prefix}/action_history_swap_divergence_mse", float(row['action_history_swap_divergence_mse']), i)
            writer.add_scalar(f"{tag_prefix}/num_pairs", float(row.get('num_pairs', 0)), i)

            # Per-slot sensitivities
            per_slot = parse_list_field(row.get('per_slot_action_sensitivity', ''))
            for si, v in enumerate(per_slot):
                writer.add_scalar(f"{tag_prefix}/per_slot/action_sensitivity_slot{si}", v, i)
            slot_imp = parse_list_field(row.get('slot_ablation_importance', ''))
            for si, v in enumerate(slot_imp):
                writer.add_scalar(f"{tag_prefix}/per_slot/slot_ablation_slot{si}", v, i)

    writer.close()
    print('Wrote TensorBoard logs to', run_dir)

if __name__ == '__main__':
    main()
