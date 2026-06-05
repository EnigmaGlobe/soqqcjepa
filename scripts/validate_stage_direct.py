import argparse, os, sys
from omegaconf import OmegaConf
import omegaconf
import torch
import pickle
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['early','mid','late'], required=True)
parser.add_argument('--ckpt', default=r'C:\Users\infra\.stable_worldmodel\local_run_bs128_ep50_weights.ckpt')
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

sys.path.insert(0, os.getcwd())
from src.train.train_causalwm_AP_node_pusht_slot import get_world_model
from src.custom_codes.custom_dataset import PushTSlotDataset
from torch.utils.data import DataLoader
import lightning as pl

# load stage pickle
stage_path = f'checkpoints/train01_slots_proj2_stage_{args.stage}.pkl'
# load base config and override
cfg_file = 'configs/config_train_causal_pusht_slot.yaml'
base_cfg = OmegaConf.load(cfg_file)
base_cfg.wandb.enable = False
base_cfg.embedding_dir = stage_path
base_cfg.action_dir = 'checkpoints/local_action_meta.pkl'
base_cfg.proprio_dir = 'checkpoints/local_proprio_meta.pkl'
base_cfg.batch_size = args.batch_size
base_cfg.num_workers = 0
# avoid loading large external pretrained weights
if 'model' in base_cfg:
    base_cfg.model.load_weights = None
with open(stage_path,'rb') as f:
    data = pickle.load(f)
# data should be {'train':{}, 'val':{video: array}}
val_slot_data = data.get('val', {})
print('val videos', list(val_slot_data.keys()))

# create val dataset
cfg = OmegaConf.create({'dinowm': {'history_size':1,'num_preds':1}, 'frameskip':1, 'seed':42, 'batch_size': args.batch_size, 'num_workers':0})
val_dataset = PushTSlotDataset(slot_data=val_slot_data, split='val', history_size=cfg.dinowm.history_size, num_preds=cfg.dinowm.num_preds, action_dir='checkpoints/local_action_meta.pkl', proprio_dir='checkpoints/local_proprio_meta.pkl', state_dir=None, frameskip=cfg.frameskip, seed=cfg.seed)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)
print('val len', len(val_dataset))

# build model
# build model using base_cfg
world_model = get_world_model(base_cfg)
trainer = pl.Trainer(accelerator='cpu', devices=1)
print('Wrapping module into LightningModule for validation...')
import torch
from types import SimpleNamespace

class _Wrapped(pl.LightningModule):
    def __init__(self, module):
        super().__init__()
        self.module = module
        try:
            self.model = module.model
        except Exception:
            self.model = None

    def validation_step(self, batch, batch_idx):
        proxy = SimpleNamespace()
        proxy.model = self.module.model
        proxy.training = False
        def _log_dict(d, on_step=True, sync_dist=True):
            try:
                self.log_dict(d, on_step=on_step, sync_dist=sync_dist)
            except Exception:
                pass
        proxy.log_dict = _log_dict
        out = self.module.forward(proxy, batch, stage='val')
        if isinstance(out, dict) and 'loss' in out:
            loss = out['loss']
            self.log('val/loss', loss)
            return loss
        return None

wrapped = _Wrapped(world_model)
print('Loading checkpoint and applying weights (non-strict)')
import torch
ckpt = torch.load(args.ckpt, map_location='cpu')
state = ckpt.get('state_dict', ckpt)
try:
    wrapped.load_state_dict(state, strict=False)
    print('Loaded checkpoint into wrapped model (non-strict)')
except Exception as e:
    print('Warning: failed to load state_dict non-strict:', e)

print('Running validate...')
res = trainer.validate(wrapped, val_loader)
print('Result', res)

# append to CSV
out_path = os.path.join('checkpoints','val_by_stage.csv')
import csv
exists = os.path.exists(out_path)
with open(out_path,'a',newline='') as fh:
    fieldnames = ['stage'] + (list(res[0].keys()) if res and isinstance(res, list) and res[0] else [])
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    if not exists:
        w.writeheader()
    row = {'stage': args.stage}
    if res and isinstance(res, list) and res[0]:
        for k,v in res[0].items():
            row[k]=float(v)
    w.writerow(row)
print('Wrote', out_path)
