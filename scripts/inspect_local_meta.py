#!/usr/bin/env python3
import pickle as pkl
import os
import numpy as np
import argparse


def summarize_map(m):
    out = {}
    for k, v in m.items():
        a = np.asarray(v)
        out[k] = (a.shape, a.dtype)
    return out


def print_sample(m, key, n=5):
    a = np.asarray(m[key])
    print(f"  sample first {n} rows for '{key}':")
    print(a[:n])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="checkpoints", help="Directory with local_*_meta.pkl files")
    args = parser.parse_args()

    paths = {
        'action': os.path.join(args.dir, 'local_action_meta.pkl'),
        'proprio': os.path.join(args.dir, 'local_proprio_meta.pkl'),
    'state': os.path.join(args.dir, 'local_state_meta.pkl'),
    'reward': os.path.join(args.dir, 'local_reward_meta.pkl'),
    }

    for name, p in paths.items():
        print(f"\n== {name} ({p}) ==")
        if not os.path.exists(p):
            print("  missing")
            continue
        with open(p, 'rb') as f:
            data = pkl.load(f)
        for split in ('train', 'val'):
            mp = data.get(split, {})
            print(f"  split '{split}': {len(mp)} videos")
            if len(mp) == 0:
                continue
            # list up to 5 keys
            keys = list(mp.keys())[:5]
            for k in keys:
                a = np.asarray(mp[k])
                print(f"    {k}: shape={a.shape}, dtype={a.dtype}")
            # show first sample for the first key
            first = keys[0]
            print_sample(mp, first, n=5)


if __name__ == '__main__':
    main()
