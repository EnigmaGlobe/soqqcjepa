#!/usr/bin/env python3
"""squeeze_slots_pickle.py

Load a slots pickle and remove a leading singleton batch dimension if present
for each video array. Saves a new pickle with suffix `_squeezed.pkl`.
"""
import argparse
import pickle as pkl
import os
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inpath", required=True)
    p.add_argument("--out", dest="outpath", required=False)
    args = p.parse_args()

    outpath = args.outpath or args.inpath.replace('.pkl', '_squeezed.pkl')
    with open(args.inpath, 'rb') as f:
        data = pkl.load(f)

    changed = False
    for split in list(data.keys()):
        for k, v in list(data[split].items()):
            arr = np.asarray(v)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr2 = arr.squeeze(0)
                data[split][k] = arr2
                changed = True

    with open(outpath, 'wb') as f:
        pkl.dump(data, f)

    print(f"Wrote {outpath}. Changed={changed}")


if __name__ == '__main__':
    main()
