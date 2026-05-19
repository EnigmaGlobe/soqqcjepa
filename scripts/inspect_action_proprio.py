import pickle as pkl
import numpy as np
import os

def inspect(pname):
    p = os.path.join('checkpoints', pname)
    print('\n==', p, '==')
    with open(p, 'rb') as f:
        d = pkl.load(f)
    for split in ('train','val'):
        m = d.get(split, {})
        print(' split', split, 'videos', len(m))
        for k, v in m.items():
            a = np.asarray(v)
            print('  ', k, 'shape', a.shape, 'nonzero', np.count_nonzero(a))

if __name__ == '__main__':
    inspect('local_action_meta.pkl')
    inspect('local_proprio_meta.pkl')
