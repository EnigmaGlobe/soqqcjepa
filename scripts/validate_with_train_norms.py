import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_mlagents_counterfactual import (
    load_world_model,
    standard_validation,
    counterfactual_action_probe,
    counterfactual_slot_ablation,
    training_stage_consistency_probe,
)
from src.custom_codes.custom_dataset import PushTSlotDataset


def load_slot_dict(pth):
    with open(pth, 'rb') as f:
        d = pickle.load(f)
    # accept either {'train':..., 'val':...} or direct mapping
    if isinstance(d, dict) and ('train' in d or 'val' in d):
        return d
    # assume whole mapping is val
    return {'val': d}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slots', required=True)
    parser.add_argument('--actions', required=True)
    parser.add_argument('--proprio', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', default='checkpoints/validation_results_trainNorm.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_batches', type=int, default=200)
    parser.add_argument('--cf_batches', type=int, default=100)
    parser.add_argument('--train_slots', default='checkpoints/train_03_slots_proj.pkl')
    parser.add_argument('--train_actions', default='checkpoints/local_action_meta.pkl')
    parser.add_argument('--train_proprio', default='checkpoints/local_proprio_meta.pkl')
    args = parser.parse_args()

    cfg = OmegaConf.load('configs/config_validate_train03_full.yaml')
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Load train stats
    train_slots = load_slot_dict(args.train_slots)
    train_slot_data = train_slots.get('train', train_slots.get('val', {}))
    train_ds = PushTSlotDataset(
        slot_data=train_slot_data,
        split='train',
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        action_dir=args.train_actions,
        proprio_dir=args.train_proprio,
        state_dir=None,
        frameskip=cfg.frameskip,
        seed=cfg.seed,
    )

    # Load val slots
    val_slots = load_slot_dict(args.slots)
    val_slot_data = val_slots.get('val', val_slots)
    val_ds = PushTSlotDataset(
        slot_data=val_slot_data,
        split='val',
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        action_dir=args.actions,
        proprio_dir=args.proprio,
        state_dir=None,
        frameskip=cfg.frameskip,
        seed=cfg.seed,
    )

    # Force train normalization onto val dataset, aligning dimensions if needed
    # Action
    try:
        val_act_w = val_ds.action_mean.shape[1]
    except Exception:
        val_act_w = None
    try:
        train_act_w = train_ds.action_mean.shape[1]
    except Exception:
        train_act_w = None
    if val_act_w is not None and train_act_w is not None:
        if train_act_w >= val_act_w:
            val_ds.action_mean = train_ds.action_mean[:, :val_act_w]
            val_ds.action_std = train_ds.action_std[:, :val_act_w]
        else:
            # pad with zeros/ones
            pad_w = val_act_w - train_act_w
            val_ds.action_mean = torch.cat([train_ds.action_mean, torch.zeros((1, pad_w))], dim=1)
            val_ds.action_std = torch.cat([train_ds.action_std, torch.ones((1, pad_w))], dim=1)

    # Proprio
    try:
        val_pro_w = val_ds.proprio_mean.shape[1]
    except Exception:
        val_pro_w = None
    try:
        train_pro_w = train_ds.proprio_mean.shape[1]
    except Exception:
        train_pro_w = None
    if val_pro_w is not None and train_pro_w is not None:
        if train_pro_w >= val_pro_w:
            val_ds.proprio_mean = train_ds.proprio_mean[:, :val_pro_w]
            val_ds.proprio_std = train_ds.proprio_std[:, :val_pro_w]
        else:
            pad_w = val_pro_w - train_pro_w
            val_ds.proprio_mean = torch.cat([train_ds.proprio_mean, torch.zeros((1, pad_w))], dim=1)
            val_ds.proprio_std = torch.cat([train_ds.proprio_std, torch.ones((1, pad_w))], dim=1)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=False)

    print(f"Loading model from {args.checkpoint}")
    world_model = load_world_model(args.checkpoint, cfg, device=device)

    print("Running standard validation (train-normalized)")
    standard = standard_validation(world_model, val_loader, cfg, device, max_batches=args.max_batches)
    print('Standard:', standard)

    print('Running counterfactual action probe')
    action_cf = counterfactual_action_probe(world_model, val_loader, cfg, device, max_batches=args.cf_batches)
    print('Action CF:', action_cf)

    print('Running slot ablation probe')
    slot_cf = counterfactual_slot_ablation(world_model, val_loader, cfg, device, max_batches=args.cf_batches)
    print('Slot CF:', slot_cf)

    print('Running training-stage consistency probe')
    stage_cf = training_stage_consistency_probe(world_model, val_loader, cfg, device, max_batches=args.cf_batches)
    print('Stage CF:', stage_cf)

    results = {
        'checkpoint': args.checkpoint,
        'slots': args.slots,
        'standard': standard,
        'counterfactual_action': action_cf,
        'counterfactual_slot_ablation': slot_cf,
        'training_stage_consistency': stage_cf,
    }
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved results to', args.out)


if __name__ == '__main__':
    main()
