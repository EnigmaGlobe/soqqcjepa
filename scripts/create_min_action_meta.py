import pickle as pkl
import numpy as np
import os

slot_path = 'checkpoints/train01_slots.pkl'
out_path = 'checkpoints/local_action_meta.pkl'
action_dim = 2  # matches configs/config_train_causal_pusht_slot.yaml

if not os.path.exists(slot_path):
    raise SystemExit(f"Missing slot embeddings: {slot_path}")

with open(slot_path, 'rb') as f:
    slots = pkl.load(f)

train_out = {}
for vid, arr in slots.get('train', {}).items():
    num_frames = int(arr.shape[0])
    train_out[vid] = np.zeros((num_frames, action_dim), dtype=np.float32)

val_out = {}
# create empty val if not present

with open(out_path, 'wb') as f:
    pkl.dump({'train': train_out, 'val': val_out}, f)

print(f'Wrote {out_path} with {len(train_out)} train entries')
