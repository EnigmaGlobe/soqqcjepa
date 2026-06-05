#!/usr/bin/env python3
import sys, os, pickle, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.utils.data import DataLoader
from src.custom_codes.custom_dataset import PushTSlotDataset

PK_SLOT='checkpoints/train01_slots.pkl'
ACT_DIR='checkpoints/local_action_meta.pkl'
PRO_DIR='checkpoints/local_proprio_meta.pkl'
STATE_DIR='checkpoints/local_state_meta.pkl'

print('Loading slot pickle:', PK_SLOT)
with open(PK_SLOT,'rb') as f:
    slot_data=pickle.load(f)

print('Train splits keys in pickle:', list(slot_data.keys()))
train_slots=slot_data.get('train')
if train_slots is None:
    print('No train split in pickle')
    sys.exit(1)

# Create dataset with small params
try:
    ds = PushTSlotDataset(
        slot_data=train_slots,
        split='train',
        history_size=1,
        num_preds=1,
        action_dir=ACT_DIR,
        proprio_dir=PRO_DIR,
        state_dir=STATE_DIR,
        frameskip=1,
        seed=42,
    )
except Exception as e:
    print('Failed to create dataset:', e)
    raise

print('Dataset length:', len(ds))
loader = DataLoader(ds, batch_size=1, num_workers=0)
try:
    batch = next(iter(loader))
except Exception as e:
    print('Failed to get one batch:', e)
    raise

import torch

def stats(t):
    if t is None:
        return 'None'
    if not isinstance(t, torch.Tensor):
        return f'NotTensor type={type(t)}'
    t=t.detach()
    flat=t.flatten().cpu().float()
    n=flat.numel()
    if n==0:
        return 'empty'
    finite= (torch.isfinite(flat)).sum().item()
    nan_ct = n - finite
    return { 'shape':tuple(t.shape), 'dtype':str(t.dtype), 'min':float(flat.min().item()), 'max':float(flat.max().item()), 'mean':float(flat.mean().item()), 'std':float(flat.std().item()), 'nan_count':int(nan_ct) }

keys = ['pixels_embed','action','proprio','embed']
for k in keys:
    v = batch.get(k, None)
    print(f'--- {k} ---')
    try:
        print(stats(v))
    except Exception as e:
        print('Error statting',k,e)

# Quick scan of action/proprio for extreme values in this batch
for k in ['action','proprio']:
    v=batch.get(k,None)
    if isinstance(v, torch.Tensor):
        flat=v.detach().float().cpu().flatten()
        print(k,'abs_max=',float(flat.abs().max().item()),'>10?', float(flat.abs().max().item())>10)

print('Done')
