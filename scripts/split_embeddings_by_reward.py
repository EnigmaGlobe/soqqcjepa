import csv, pickle, os, sys
from collections import defaultdict
import numpy as np

obs_csv = sys.argv[1] if len(sys.argv)>1 else 'testdata/validation/3/data_training/observations_frame_train_03.csv'
emb_path = sys.argv[2] if len(sys.argv)>2 else 'checkpoints/train01_slots_proj2.pkl'
out_dir = sys.argv[3] if len(sys.argv)>3 else 'checkpoints'

print('Reading observations:', obs_csv)
frame_rewards = defaultdict(list)
with open(obs_csv, newline='') as fh:
    r = csv.DictReader(fh)
    for row in r:
        try:
            fc = int(row.get('frame_count','0'))
        except:
            fc = 0
        try:
            rew = float(row.get('reward','0'))
        except:
            rew = 0.0
        frame_rewards[fc].append(rew)

# Compute mean reward per frame index up to max_frame
if not frame_rewards:
    raise SystemExit('No frame rewards found')
max_frame = max(frame_rewards.keys())
print('Max frame index found:', max_frame)
mean_rewards = np.zeros(max_frame+1, dtype=float)
for i in range(max_frame+1):
    vals = frame_rewards.get(i, [])
    mean_rewards[i] = float(np.mean(vals)) if vals else 0.0

# compute tertiles
p33 = np.percentile(mean_rewards, 33)
p66 = np.percentile(mean_rewards, 66)
print('Percentiles: 33%=', p33, '66%=', p66)

# load embeddings
print('Loading embeddings:', emb_path)
with open(emb_path, 'rb') as f:
    emb = pickle.load(f)
# emb expected shape [num_frames, num_slots, dim]
import numpy as _np
emb = _np.asarray(emb)
num_frames = emb.shape[0]
print('Embeddings shape:', emb.shape)

# if embedding frames != mean_rewards length, align by min
L = min(num_frames, mean_rewards.shape[0])
print('Using first', L, 'frames for splitting')
mean_rewards = mean_rewards[:L]
emb = emb[:L]

masks = {
    'early': mean_rewards <= p33,
    'mid': (mean_rewards > p33) & (mean_rewards <= p66),
    'late': mean_rewards > p66,
}

os.makedirs(out_dir, exist_ok=True)
outs = {}
for k,mask in masks.items():
    idx = np.nonzero(mask)[0]
    print(f'{k}: {len(idx)} frames')
    outp = os.path.join(out_dir, f'train01_slots_proj2_stage_{k}.pkl')
    with open(outp, 'wb') as f:
        pickle.dump(emb[idx], f)
    outs[k] = outp

print('Wrote:', outs)
