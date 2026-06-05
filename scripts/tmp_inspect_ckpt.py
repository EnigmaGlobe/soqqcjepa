import torch
import json
import sys

ckpt_path = r"C:\Users\infra\.stable_worldmodel\local_run_bs128_ep50_weights.ckpt"
print('Loading', ckpt_path)
ck = torch.load(ckpt_path, map_location='cpu')
sd = ck.state_dict() if hasattr(ck, 'state_dict') else (ck.get('state_dict', ck) if isinstance(ck, dict) else ck)

keys = sorted(list(sd.keys()))
print(f'total keys: {len(keys)}')
sample = [k for k in keys if ('action' in k or 'proprio' in k or 'proprio' in k)]
print('sample action/proprio keys (first 50):')
print('\n'.join(sample[:50]))

shapes = {}
for k in sample:
    try:
        shapes[k] = tuple(sd[k].shape)
    except Exception:
        shapes[k] = str(type(sd[k]))

print('\nshapes:')
print(json.dumps(shapes, indent=2))

# Check predictor input dim inference
slot_dim = None
for k in keys:
    if 'videosaur.SLOT_DIM' in k:
        pass

print('\nDone')
