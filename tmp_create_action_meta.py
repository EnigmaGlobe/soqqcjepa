import pickle as pkl
import numpy as np
from pathlib import Path
p=Path('checkpoints')/ 'train01_slots.pkl'
out=Path('checkpoints')/ 'local_action_meta.pkl'
print('Loading',p)
slots = pkl.load(open(p,'rb'))
meta = {}
for split in ['train','val']:
    meta[split] = {}
    if split in slots and isinstance(slots[split], dict):
        for vid, arr in slots[split].items():
            num_frames = int(arr.shape[0])
            meta[split][vid] = np.zeros((num_frames,1), dtype=np.float32)
print('Writing', out)
pkl.dump(meta, open(out,'wb'))
print('Done')
