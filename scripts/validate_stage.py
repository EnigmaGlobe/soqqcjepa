import argparse
from omegaconf import OmegaConf
import torch
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['early','mid','late'], required=True)
parser.add_argument('--ckpt', default=r'C:\Users\infra\.stable_worldmodel\local_run_bs128_ep50_weights.ckpt')
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

# build minimal config matching train script
cfg = OmegaConf.create({
    'cache_dir': 'checkpoints',
    'wandb': {'project':'local','entity':None},
    'output_model_name':'local_run_bs128_ep50',
    'trainer': {'max_epochs':1,'accelerator':'cpu','devices':1},
    'seed':42,
    'embedding_dir': f'checkpoints/train01_slots_proj2_stage_{args.stage}.pkl',
    'action_dir': 'checkpoints/local_action_meta.pkl',
    'proprio_dir': 'checkpoints/local_proprio_meta.pkl',
    'state_dir': None,
    'dinowm': {'history_size':1,'num_preds':1},
    'batch_size': args.batch_size,
    'num_workers': 0,
    'frameskip':1,
})

# import local helpers
sys.path.insert(0, os.getcwd())
from src.train.train_causalwm_AP_node_pusht_slot import get_data, get_world_model
import lightning as pl

print('Loading data...')
data = get_data(cfg)
print('Building model...')
world_model = get_world_model(cfg)

trainer = pl.Trainer(**cfg.trainer)
print('Validating stage', args.stage, 'using ckpt', args.ckpt)
res = trainer.validate(world_model, datamodule=data, ckpt_path=args.ckpt)
print('Result:', res)

# write CSV line
out_path = os.path.join('checkpoints', 'val_by_stage.csv')
import csv
fieldnames = ['stage'] + (list(res[0].keys()) if res and isinstance(res, list) and res[0] else [])
exists = os.path.exists(out_path)
with open(out_path,'a',newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    if not exists:
        w.writeheader()
    row = {'stage': args.stage}
    if res and isinstance(res, list) and res[0]:
        for k,v in res[0].items():
            row[k] = float(v)
    w.writerow(row)
print('Wrote', out_path)
