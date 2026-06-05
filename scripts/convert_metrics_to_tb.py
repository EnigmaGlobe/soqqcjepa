#!/usr/bin/env python3
import os
import csv
from torch.utils.tensorboard import SummaryWriter

ROOT = "lightning_logs"
OUT_ROOT = "lightning_logs_tb"

os.makedirs(OUT_ROOT, exist_ok=True)

for version in sorted(os.listdir(ROOT)):
    version_path = os.path.join(ROOT, version)
    csv_path = os.path.join(version_path, "metrics.csv")
    if not os.path.isfile(csv_path):
        continue
    out_path = os.path.join(OUT_ROOT, version)
    os.makedirs(out_path, exist_ok=True)
    writer = SummaryWriter(log_dir=out_path)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        step_counter = 0
        for row in reader:
            # Determine a sensible step index
            if 'step' in row and row['step']:
                try:
                    step = int(float(row['step']))
                except Exception:
                    step_counter += 1
                    step = step_counter
            elif 'epoch' in row and row['epoch']:
                try:
                    step = int(float(row['epoch']))
                except Exception:
                    step_counter += 1
                    step = step_counter
            else:
                step_counter += 1
                step = step_counter

            for k, v in row.items():
                if k.lower() in ('epoch', 'step', 'time', 'timestamp'):
                    continue
                try:
                    val = float(v)
                except Exception:
                    continue
                writer.add_scalar(k, val, global_step=step)
    writer.close()
    print(f"Wrote TB events to {out_path}")

print("Conversion complete.")
