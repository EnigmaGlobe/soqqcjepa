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
