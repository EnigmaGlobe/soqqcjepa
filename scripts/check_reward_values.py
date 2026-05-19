#!/usr/bin/env python3
import csv
import os
import argparse


def parse_reward(s):
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-dir", required=True)
    args = p.parse_args()

    csv_files = [os.path.join(args.csv_dir, f) for f in os.listdir(args.csv_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        print("No CSV files found in", args.csv_dir); return

    for csvf in csv_files:
        print('\n==', os.path.basename(csvf), '==')
        with open(csvf, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            print('fields:', fields)
            # find reward-like column names
            reward_cols = [c for c in fields if 'reward' in c.lower()]
            if not reward_cols:
                print('  no "reward" column found (case-insensitive)')
            else:
                for rc in reward_cols:
                    nonzeros = []
                    cnt = 0
                    total = 0
                    for row in reader:
                        total += 1
                        r = parse_reward(row.get(rc))
                        if r is not None and r != 0.0:
                            cnt += 1
                            if len(nonzeros) < 10:
                                ep = row.get('episode_id') or row.get('episode') or row.get('episode_idx') or ''
                                step = row.get('step_index') or row.get('step') or row.get('step_idx') or row.get('frame') or ''
                                nonzeros.append((total, ep, step, r))
                    print(f"  reward column '{rc}': total rows={total}, non-zero count={cnt}")
                    if nonzeros:
                        print('   examples (row#, episode, step, reward):')
                        for ex in nonzeros:
                            print('    ', ex)
                    # reset reader to start for next column
                    fh.seek(0)
                    reader = csv.DictReader(fh)


if __name__ == '__main__':
    main()
