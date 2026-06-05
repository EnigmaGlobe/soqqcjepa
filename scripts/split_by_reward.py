import csv, os, pickle, numpy as np
from collections import defaultdict

csv_path = 'testdata/validation/3/data_training/observations_frame_train_03.csv'
embed_path = 'checkpoints/train01_slots_proj2.pkl'
out_dir = 'checkpoints'

# 1) accumulate rewards per frame_count
frame_rewards = defaultdict(list)
count = 0
with open(csv_path, newline='') as fh:
    reader = csv.DictReader(fh)
    for r in reader:
        count += 1
        try:
            fc = int(r.get('frame_count',''))
            reward = float(r.get('reward','0'))
        except Exception:
            continue
        frame_rewards[fc].append(reward)

if not frame_rewards:
    raise SystemExit('no frame rewards found')

max_fc = max(frame_rewards.keys())
print('frames with rewards', len(frame_rewards), 'max frame_count', max_fc)

# build mean reward per frame index 0..max_fc
rewards = np.zeros(max_fc+1, dtype=float)
for i in range(max_fc+1):
    vals = frame_rewards.get(i, [0.0])
    rewards[i] = float(np.mean(vals))

# load embeddings
with open(embed_path,'rb') as f:
    data = pickle.load(f)
train_key = list(data['train'].keys())[0]
frames = np.array(data['train'][train_key])
print('embeddings frames', frames.shape)

num_frames = frames.shape[0]
# truncate or pad rewards to num_frames
if rewards.shape[0] < num_frames:
    # pad with last value
    pad = np.full((num_frames - rewards.shape[0],), rewards[-1] if rewards.size>0 else 0.0)
    rewards = np.concatenate([rewards, pad])
elif rewards.shape[0] > num_frames:
    rewards = rewards[:num_frames]

print('aligned rewards length', rewards.shape)

# compute smoothed rewards (rolling mean) to capture learning trend
win = min(1000, len(rewards))
smoothed = np.convolve(rewards, np.ones(win)/win, mode='same')
p33 = np.percentile(smoothed, 33.3333)
p66 = np.percentile(smoothed, 66.6667)
print('percentiles (smoothed)', p33, p66)

early_idx = np.where(smoothed <= p33)[0]
mid_idx = np.where((smoothed > p33) & (smoothed <= p66))[0]
late_idx = np.where(smoothed > p66)[0]
print('counts (smoothed)', len(early_idx), len(mid_idx), len(late_idx))

# build stage pickles: keep train empty, put selected frames under val with same key name
os.makedirs(out_dir, exist_ok=True)
for name, idxs in [('early', early_idx), ('mid', mid_idx), ('late', late_idx)]:
    out = {'train': {}, 'val': {}}
    # keep train empty
    sel = frames[idxs]
    out['val'][train_key] = sel
    out_path = os.path.join(out_dir, f'train01_slots_proj2_stage_{name}.pkl')
    with open(out_path, 'wb') as of:
        pickle.dump(out, of)
    print('wrote', out_path, 'shape', sel.shape)

# also save rewards per stage summary
with open(os.path.join(out_dir, 'reward_stage_summary.txt'),'w') as fh:
    fh.write(f'p33={p33}, p66={p66}\n')
    fh.write(f'counts early={len(early_idx)}, mid={len(mid_idx)}, late={len(late_idx)}\n')

print('done')
