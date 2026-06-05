"""Run real training bypassing hydra by calling the underlying run() with a constructed cfg.

This script monkeypatches the Videosaur `models.build` to return lightweight placeholders
so we don't depend on DINO/large encoder during this smoke training.

It runs 1 epoch, batch_size=1, CPU.
"""
import sys
from types import SimpleNamespace
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import importlib

sys.path.insert(0, r"C:\soqqle\soqqcjepa")

# Monkeypatch videosaur.models.build to return a simple dummy object
try:
    from src.third_party.videosaur.videosaur import models as vs_models
    class DummyInit(nn.Module):
        def __init__(self, n_slots=4, dim=128):
            super().__init__()
            self.n_slots = n_slots
            self.dim = dim
        def forward(self, batch_size=1):
            return torch.zeros((batch_size, self.n_slots, self.dim))

    class DummyModule:
        def __init__(self, n_slots=4, dim=128):
            self.encoder = nn.Identity()
            self.processor = nn.Identity()
            self.initializer = DummyInit(n_slots=n_slots, dim=dim)

    def _build_dummy(*args, **kwargs):
        # infer n_slots and dim from kwargs or fallback
        n = kwargs.get('n_slots', 4)
        d = kwargs.get('dim', 128)
        return DummyModule(n_slots=n, dim=d)

    vs_models.build = _build_dummy
except Exception as e:
    print('Warning: failed to monkeypatch videosaur.models.build:', e)

# Load training module
mod = importlib.import_module('src.train.train_causalwm_AP_node_pusht_slot')

# Build a minimal cfg expected by the training script
cfg = OmegaConf.create({
    'cache_dir': 'checkpoints',
    'wandb': {'enable': False, 'project': 'local', 'entity': None},
    'output_model_name': 'local_real_train',
    'trainer': {'max_epochs': 1, 'accelerator': 'cpu', 'devices': 1, 'log_every_n_steps': 1},
    'seed': 42,
    'embedding_dir': 'checkpoints/train01_slots.pkl',
    'action_dir': 'checkpoints/local_action_meta.pkl',
    'proprio_dir': 'checkpoints/local_proprio_meta.pkl',
    'state_dir': 'checkpoints/local_state_meta.pkl',
    'frameskip': 1,
    'batch_size': 1,
    'num_workers': 0,
    'predictor_lr': 5e-4,
    'proprio_encoder_lr': 5e-4,
    'action_encoder_lr': 5e-4,
    'dinowm': {'history_size': 1, 'num_preds': 1, 'proprio_dim': 7, 'action_dim': 2},
    'predictor': {'depth': 1, 'heads': 4, 'mlp_dim': 256, 'dim_head': 64, 'dropout': 0.1},
    'videosaur': {'NUM_SLOTS': 4, 'SLOT_DIM': 128, 'FEAT_DIM': 384},
    'model': {},
    'dummy_optimizer': {'name': 'Adam', 'lr': 1e-3}
})

# Call the underlying function (bypass hydra decorator if present)
run_fn = getattr(mod.run, '__wrapped__', mod.run)

print('Starting training run (1 epoch, CPU)')
try:
    run_fn(cfg)
    print('Training run finished')
except Exception as e:
    print('Primary training path failed, falling back to simple trainer:', e)

    # Fallback: construct data and a lightweight LightningModule to train predictor + encoders
    from src.train.train_causalwm_AP_node_pusht_slot import get_data
    data = get_data(cfg)
    train_loader = data.train

    # Build a lightweight LightningModule
    import torch.nn as nn
    import torch
    import pytorch_lightning as pl
    from src.cjepa_predictor import MaskedSlot_AP_Predictor

    class QuickWM(pl.LightningModule):
        def __init__(self, cfg):
            super().__init__()
            S = int(cfg.videosaur.NUM_SLOTS)
            D = int(cfg.videosaur.SLOT_DIM)
            self.history = int(cfg.dinowm.history_size)
            self.num_preds = int(cfg.dinowm.num_preds)
            self.predictor = MaskedSlot_AP_Predictor(
                num_slots=S + 2,
                slot_dim=D,
                history_frames=self.history,
                pred_frames=self.num_preds,
                num_masked_slots=int(cfg.get('num_masked_slots', 2)),
                depth=int(cfg.predictor.depth),
                heads=int(cfg.predictor.heads),
                dim_head=int(cfg.predictor.dim_head),
                mlp_dim=int(cfg.predictor.mlp_dim),
                dropout=float(cfg.predictor.dropout),
            )
            # simple linear encoders
            act_dim = int(cfg.dinowm.action_dim) * int(cfg.frameskip)
            prop_dim = int(cfg.dinowm.proprio_dim)
            self.action_encoder = nn.Linear(act_dim, D)
            self.proprio_encoder = nn.Linear(prop_dim, D)

        def forward(self, batch):
            pixels = batch['pixels_embed'].float()  # (B, T, S, D)
            B, T, S, D = pixels.shape
            embedding = pixels
            if 'proprio' in batch:
                proprio = batch['proprio'].float()
                proprio_embed = self.proprio_encoder(proprio)
                proprio_tiled = proprio_embed.unsqueeze(2)
                embedding = torch.cat([embedding, proprio_tiled], dim=2)
            if 'action' in batch:
                action = batch['action'].float()
                action_embed = self.action_encoder(action)
                action_tiled = action_embed.unsqueeze(2)
                embedding = torch.cat([embedding, action_tiled], dim=2)

            # predictor expects (B, T_hist, S_total, D)
            hist = embedding[:, : self.history]
            pred_out = self.predictor(hist)
            if isinstance(pred_out, tuple):
                pred_embedding, mask_indices = pred_out
            else:
                pred_embedding = pred_out
                mask_indices = []

            # compute simple MSE loss between pred_future and target
            target = embedding[:, self.history : self.history + self.num_preds, :S, :].detach()
            pred_future = pred_embedding[:, self.history : self.history + self.num_preds, :S, :]
            loss = nn.functional.mse_loss(pred_future, target)
            return loss

        def training_step(self, batch, batch_idx):
            loss = self.forward(batch)
            self.log('train/loss', loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    model = QuickWM(cfg)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model, train_dataloaders=train_loader)
    print('Fallback training finished')
