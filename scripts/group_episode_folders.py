#!/usr/bin/env python
"""Move episode folders into skill groups.

Default grouping used for this dataset:
- exploring: episode_1 .. episode_160
- well_trained: episode_500 .. end

Episodes between those ranges are left in place as transition episodes.
"""
from pathlib import Path
import argparse
import shutil


def move_range(source_dir, dest_dir, start_ep, end_ep=None):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    missing = 0
    for episode_dir in sorted(source_dir.glob('episode_*')):
        try:
            episode_num = int(episode_dir.name.split('_', 1)[1])
        except Exception:
            continue

        if episode_num < start_ep:
            continue
        if end_ep is not None and episode_num > end_ep:
            continue

        target = dest_dir / episode_dir.name
        if target.exists():
            continue

        if episode_dir.exists():
            shutil.move(str(episode_dir), str(target))
            moved += 1
        else:
            missing += 1

    return moved, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', required=True)
    parser.add_argument('--dest_root', required=True)
    parser.add_argument('--exploring_end', type=int, default=160)
    parser.add_argument('--well_trained_start', type=int, default=500)
    args = parser.parse_args()

    dest_root = Path(args.dest_root)
    exploring_dir = dest_root / 'exploring'
    well_trained_dir = dest_root / 'well_trained'

    moved_exploring, _ = move_range(args.source_dir, exploring_dir, 1, args.exploring_end)
    moved_well_trained, _ = move_range(args.source_dir, well_trained_dir, args.well_trained_start, None)

    print(f'Moved {moved_exploring} exploring episode folders to {exploring_dir}')
    print(f'Moved {moved_well_trained} well-trained episode folders to {well_trained_dir}')
    print('Transition episodes were left in the source directory.')


if __name__ == '__main__':
    main()
