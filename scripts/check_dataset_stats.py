from src.custom_codes.custom_dataset import PushTSlotDataset
import pickle
import torch

slot_p = 'checkpoints/train01_slots.pkl'
action_p = 'checkpoints/local_action_meta.pkl'
proprio_p = 'checkpoints/local_proprio_meta.pkl'
state_p = 'checkpoints/local_state_meta.pkl'

slot_data = pickle.load(open(slot_p,'rb'))['train']

ds = PushTSlotDataset(
    slot_data=slot_data,
    split='train',
    history_size=5,
    num_preds=3,
    action_dir=action_p,
    proprio_dir=proprio_p,
    state_dir=state_p,
    frameskip=3,
    seed=42,
)

print('n_samples', len(ds))
print('action_mean', getattr(ds,'action_mean',None))
print('action_std', getattr(ds,'action_std',None))
print('action_std any zero', (ds.action_std==0).any())
print('action_std any nan', torch.isnan(ds.action_std).any())
print('proprio_std any zero', (ds.proprio_std==0).any())
print('proprio_std any nan', torch.isnan(ds.proprio_std).any())

try:
    # sample a batch
    s = ds[0]
    for k,v in s.items():
        print(k, type(v), getattr(v,'shape',None), torch.isfinite(v).all())
except Exception as e:
    print('sample err', e)
