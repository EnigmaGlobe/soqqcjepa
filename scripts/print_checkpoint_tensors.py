import sys
import torch
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/local_real_train_fp32_logs_foreground_object.ckpt'
print('Loading', path)
obj = torch.load(path, map_location='cpu')
print('Loaded type:', type(obj))

def inspect_state(d, max_items=200):
    if not isinstance(d, dict):
        print('State is not a dict')
        return
    print('Num keys:', len(d))
    keys = list(d.keys())
    for k in keys[:max_items]:
        v = d[k]
        try:
            if hasattr(v, 'shape'):
                print(k, 'shape=', getattr(v, 'shape', None), 'dtype=', getattr(v, 'dtype', None))
            else:
                print(k, type(v))
        except Exception as e:
            print(k, 'inspect-error', e)

if isinstance(obj, dict):
    inspect_state(obj)
else:
    # try to get state_dict()
    try:
        sd = obj.state_dict()
        print('object.state_dict() found')
        inspect_state(sd)
    except Exception as e:
        print('Not a dict and no state_dict():', e)
        # try to introspect attributes
        attrs = [a for a in dir(obj) if not a.startswith('_')]
        print('Attributes:', attrs[:50])
        # try to find a model member
        if hasattr(obj, 'model'):
            print('Has .model, inspecting model.state_dict()')
            try:
                inspect_state(obj.model.state_dict())
            except Exception as e2:
                print('Failed to inspect obj.model.state_dict():', e2)

print('Done')
