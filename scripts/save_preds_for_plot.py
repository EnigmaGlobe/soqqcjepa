import argparse
import numpy as np
import torch
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.validate_mlagents_counterfactual as vmc
from omegaconf import OmegaConf


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='outputs/tmp_val_examples.npz')
    p.add_argument('--max_batches', type=int, default=6)
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wm = vmc.load_world_model(args.checkpoint, cfg, device=device)
    val_loader = vmc.build_val_loader(cfg)

    preds_list = []
    targs_list = []

    slot_dim = cfg.videosaur.SLOT_DIM
    for batch_idx, batch in enumerate(val_loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        slots = batch['pixels_embed'].to(device)
        actions = batch['action'].to(device) if 'action' in batch else None
        proprios = batch['proprio'].to(device) if 'proprio' in batch else None
        B, T, S, D = slots.shape

        emb = vmc._build_embedding(wm, slots, actions, proprios, S)
        hist = emb[:, : cfg.dinowm.history_size, :, :]
        pred = wm.predict(hist, use_inference_function=True)[..., :slot_dim]

        tgt = emb[:, cfg.dinowm.history_size : cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :][..., :slot_dim]

        preds_list.append(pred.detach().cpu().numpy())
        targs_list.append(tgt.detach().cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    targs = np.concatenate(targs_list, axis=0)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, preds=preds, targs=targs)
    print('Saved preds/targs to', args.output)


if __name__ == '__main__':
    main()
