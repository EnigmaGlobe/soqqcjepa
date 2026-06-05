import pickle
import torch
import numpy as np
import pandas as pd
import sys


def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        try:
            return torch.load(path, map_location='cpu')
        except Exception as e:
            print('load error:', e)
            return None


def chk_arr(a, name):
    try:
        arr = np.array(a)
        print(f"{name}: shape={arr.shape}, nan={np.isnan(arr).any()}, inf={np.isinf(arr).any()}, mean={np.nanmean(arr)}, std={np.nanstd(arr)}")
    except Exception as e:
        print(f"err checking {name}: {e}")


if __name__ == '__main__':
    pkl = 'checkpoints/train01_slots.pkl'
    print('Checking pickle:', pkl)
    obj = load_pickle(pkl)
    print('type:', type(obj))
    if obj is None:
        print('Could not load pickle')
    else:
        if isinstance(obj, dict):
            for k, v in obj.items():
                chk_arr(v, f'pickle[{k}]')
        else:
            chk_arr(obj, 'pickle')

    csv = 'testdata/train01/actions_frame_run_04_train_01.csv'
    print('\nChecking CSV (first 100k rows):', csv)
    try:
        df = pd.read_csv(csv, nrows=100000)
        print('csv shape:', df.shape)
        print('nulls per column (sample):')
        print(df.isna().sum().head(20).to_dict())
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] > 0:
            print('numeric cols count:', num.shape[1])
            print('any inf in numeric?', np.isinf(num.values).any())
            desc = num.describe().loc[['min', 'max', 'mean']]
            print('numeric stats (min/max/mean) sample:')
            print(desc.head(10).to_dict())
        else:
            print('No numeric columns found in CSV sample')
    except Exception as e:
        print('CSV read error:', e)

    # Inspect action/proprio pickles used by dataset
    for path in ['checkpoints/local_action_meta.pkl', 'checkpoints/local_proprio_meta.pkl', 'checkpoints/local_state_meta.pkl']:
        print('\nChecking pickle:', path)
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print('  load error:', e)
            continue
        print('  type:', type(data), 'videos:', len(getattr(data, 'keys', lambda: [])()))
        # inspect up to first 3 videos
        for i, (k, v) in enumerate(list(data.items())[:3]):
            try:
                print(f'  video {k}: type={type(v)}, len={getattr(v, "shape", getattr(v, "__len__", None))}')
                # print types of first elements
                try:
                    sample = v[:5]
                except Exception:
                    sample = list(v)[:5]
                print('   sample element types:', [type(x) for x in sample])
                # if numeric arrays inside, print their shapes
                for j, x in enumerate(sample):
                    try:
                        ax = np.asarray(x)
                        print(f'    elem {j} shape {ax.shape} dtype {ax.dtype} min/max {np.nanmin(ax)}/{np.nanmax(ax)}')
                    except Exception as e:
                        print(f'    elem {j} inspect err:', e)
            except Exception as e:
                print('   err inspecting video', k, e)

    # Deeper inspect: show sample value types for train split
    for path in ['checkpoints/local_action_meta.pkl', 'checkpoints/local_proprio_meta.pkl']:
        print('\nDeep inspect:', path)
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
        except Exception as e:
            print('  load err', e)
            continue
        if 'train' in obj:
            d = obj['train']
            print(' train len', len(d), 'sample keys', list(d.keys())[:5])
            for k in list(d.keys())[:3]:
                v = d[k]
                print('  key', k, 'value type', type(v))
                try:
                    # if dict, show nested keys
                    if isinstance(v, dict):
                        print('    nested dict keys sample', list(v.keys())[:5])
                    else:
                        a = np.asarray(v)
                        print('    arr shape', a.shape, 'dtype', a.dtype)
                except Exception as e:
                    print('    inspect err', e)
