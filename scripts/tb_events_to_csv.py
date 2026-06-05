#!/usr/bin/env python3
import os
import csv
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "lightning_logs"
START = 25
if len(sys.argv) > 1:
    try:
        START = int(sys.argv[1])
    except Exception:
        pass

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
    ev_files = [f for f in os.listdir(version_path) if f.startswith('events.out.tfevents')]
    if not ev_files:
        print(f"Skipping {version}: no event files")
        continue
    ev_path = os.path.join(version_path, ev_files[0])
    try:
        ea = EventAccumulator(ev_path, size_guidance={'scalars': 0})
        ea.Reload()
    except Exception as e:
        print(version, 'FAILED to load event file:', e)
        continue

    tags = ea.Tags().get('scalars', [])
    if not tags:
        print(f"{version}: no scalar tags")
        continue

    out_csv = os.path.join(version_path, 'metrics.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tag', 'wall_time', 'step', 'value'])
        for tag in tags:
            try:
                vals = ea.Scalars(tag)
            except Exception:
                continue
            for ev in vals:
                writer.writerow([tag, ev.wall_time, ev.step, ev.value])
    print(f"Wrote {out_csv}")

print('Done')
