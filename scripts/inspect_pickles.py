import pickle
import os
import json

def inspect(path, max_items=5):
    print('---', path)
    if not os.path.exists(path):
        print(' MISSING')
        return
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print('type:', type(obj))
    try:
        keys = list(obj.keys())
        print(' top keys (sample):', keys[:10])
    except Exception:
        print(' not a mapping')
    # if it's a dict with 'train'/'val'
    if isinstance(obj, dict) and 'train' in obj:
        d = obj['train']
        print(' train keys count:', len(d))
        for i, k in enumerate(list(d.keys())[:max_items]):
            v = d[k]
            print(f'  key[{i}] = {k!r} -> type {type(v)}')
            try:
                # if array-like
                import numpy as np
                a = np.asarray(v)
                print('    asarray shape', a.shape, 'dtype', a.dtype)
                # print a small sample
                if a.size > 0:
                    print('    sample[0] repr:', repr(a.flatten()[0])[:200])
            except Exception as e:
                print('    inspect err:', e)
    else:
        # show small repr if possible
        try:
            print(' repr sample:', repr(obj)[:500])
        except Exception:
            pass


if __name__ == '__main__':
    inspect('checkpoints/train01_slots.pkl')
    inspect('checkpoints/local_action_meta.pkl')
    inspect('checkpoints/local_proprio_meta.pkl')
    inspect('checkpoints/local_state_meta.pkl')
import pickle
from pathlib import Path

def inspect(p):
    d = pickle.load(open(p,'rb'))
    print(p.name, 'top keys:', list(d.keys()))
    if isinstance(d, dict) and 'train' in d:
        print(' train keys:', list(d['train'].keys()))
        if len(d['train']):
            k, v = next(iter(d['train'].items()))
            import numpy as np
            print(' sample shape:', np.array(v).shape)
    print()

for name in ['movie_first3_slots.pkl','action_meta.pkl','proprio_meta.pkl','state_meta.pkl']:
    p = Path('checkpoints') / name
    inspect(p)
