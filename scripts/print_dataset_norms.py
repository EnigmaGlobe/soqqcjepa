"""Print action/proprio normalization stats for train and val splits.

Usage:
  python scripts/print_dataset_norms.py --slots checkpoints/train_03_slots_proj.pkl --actions checkpoints/local_action_meta.pkl --proprio checkpoints/local_proprio_meta.pkl
"""
import argparse
import pickle
import torch
from src.custom_codes.custom_dataset import PushTSlotDataset

parser = argparse.ArgumentParser()
parser.add_argument('--slots', default='checkpoints/train_03_slots_proj.pkl')
parser.add_argument('--actions', default='checkpoints/local_action_meta.pkl')
parser.add_argument('--proprio', default='checkpoints/local_proprio_meta.pkl')
parser.add_argument('--history', type=int, default=1)
parser.add_argument('--preds', type=int, default=1)
parser.add_argument('--frameskip', type=int, default=1)
args = parser.parse_args()

with open(args.slots,'rb') as f:
    slots = pickle.load(f)

for split in ('train','val'):
    slot_data = slots.get(split, {})
    if not slot_data:
        print(f"No slot data for split {split}")
        continue
    ds = PushTSlotDataset(slot_data=slot_data, split=split, history_size=args.history, num_preds=args.preds, action_dir=args.actions, proprio_dir=args.proprio, state_dir=None, frameskip=args.frameskip, seed=42)
    print(f"--- {split} ---")
    print('n_samples', len(ds))
    print('action_mean', getattr(ds,'action_mean',None))
    print('action_std', getattr(ds,'action_std',None))
    print('proprio_mean', getattr(ds,'proprio_mean',None))
    print('proprio_std', getattr(ds,'proprio_std',None))
    print()
print('Done')
