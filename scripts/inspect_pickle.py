import pickle, os, numpy as np
p='checkpoints/train01_slots_proj2.pkl'
print('exists', os.path.exists(p))
data = pickle.load(open(p,'rb'))
print('type', type(data))
if isinstance(data, dict):
    for k, v in data.items():
        print('SECTION', k, 'type', type(v))
        try:
            if isinstance(v, dict):
                print('  keys:', list(v.keys())[:10], '...')
                for kk in list(v.keys())[:3]:
                    print('   key', kk, '->', type(v[kk]), 'len', len(v[kk]) if hasattr(v[kk], '__len__') else 'n/a')
            else:
                arr = np.array(v)
                print('  array shape', arr.shape, arr.dtype)
        except Exception as e:
            print('  inspect error', e)
else:
    arr = np.array(data)
    print('shape', arr.shape, arr.dtype)
