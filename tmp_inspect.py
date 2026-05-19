import pickle
import os

pfn = os.path.join('checkpoints','train01_slots.pkl')
print('file:', pfn, 'exists=', os.path.exists(pfn))
try:
    p = pickle.load(open(pfn,'rb'))
except Exception as e:
    print('load error:', e)
    raise

print('top-type:', type(p))
if isinstance(p, dict):
    for k,v in p.items():
        print('TOP KEY', k, '->', type(v))
        if isinstance(v, dict):
            print('  subkeys count', len(v))
            for kk, arr in list(v.items())[:10]:
                try:
                    print('   ', kk, 'shape=', getattr(arr, 'shape', None), 'type=', type(arr))
                except Exception as e:
                    print('   ', kk, 'error', e)
        else:
            print('  value shape=', getattr(v, 'shape', None), 'type=', type(v))
else:
    print('pickle content non-dict, repr:', repr(p)[:200])
