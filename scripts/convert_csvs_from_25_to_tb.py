#!/usr/bin/env python3
import os
import csv
import sys
from torch.utils.tensorboard import SummaryWriter

ROOT = "lightning_logs"
OUT_ROOT = "lightning_logs_tb"
START = 25
if len(sys.argv) > 1:
    try:
        START = int(sys.argv[1])
    except Exception:
        pass

os.makedirs(OUT_ROOT, exist_ok=True)

for version in sorted(os.listdir(ROOT)):
    if not version.startswith('version_'):
        continue
    try:
        idx = int(version.split('_')[1])
    except Exception:
        continue
    if idx < START:
        continue
    version_path = os.path.join(ROOT, version)
    csv_path = os.path.join(version_path, "metrics.csv")
    if not os.path.isfile(csv_path):
        print(f"Skipping {version}: no metrics.csv")
        continue
    out_path = os.path.join(OUT_ROOT, version)
    os.makedirs(out_path, exist_ok=True)
    writer = SummaryWriter(log_dir=out_path)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        # Detect long format produced by tb_events_to_csv.py: columns = ['tag','wall_time','step','value']
        cols = reader.fieldnames or []
        if 'tag' in cols and 'value' in cols:
            # long format: each row has a tag and a value
            for row in reader:
                try:
                    step = int(float(row.get('step') or 0))
                except Exception:
                    step = None
                tag = row.get('tag')
                try:
                    val = float(row.get('value'))
                except Exception:
                    continue
                if step is None:
                    writer.add_scalar(tag, val)
                else:
                    writer.add_scalar(tag, val, global_step=step)
        else:
            # wide format: columns are metric names
            step_counter = 0
            for row in reader:
                if 'step' in row and row['step']:
                    try:
                        step = int(float(row['step']))
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
