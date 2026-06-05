from pathlib import Path

# Hydration: try to import hydra; provide a safe fallback for environments
# where hydra is unavailable or incompatible (e.g., Python 3.14 issues).
try:
    import hydra
except Exception:
    import types

    def _hydra_main(*args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    hydra = types.SimpleNamespace(main=_hydra_main)
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torchvision
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import Dataset, DataLoader
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.nn import functional as F
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from transformers import AutoModel
import wandb
from src.world_models.dinowm_causal_AP_node import CausalWM_AP
from src.cjepa_predictor import MaskedSlot_AP_Predictor
from src.third_party.videosaur.videosaur import models
from src.custom_codes.hungarian import hungarian_matching_loss_AP, hungarian_matching_loss_with_proprio
from src.custom_codes.custom_dataset import PushTSlotDataset

import pickle as pkl
import numpy as np

import os




# ============================================================================
# Data Setup
# ============================================================================
def get_data(cfg):
    """Setup dataset with pre-extracted slot representations."""
    
    # Load pre-extracted slot embeddings
    with open(cfg.embedding_dir, "rb") as f:
        slot_data = pkl.load(f)
    
    logging.info(f"Loaded slot embeddings from {cfg.embedding_dir}")
    
    train_dataset = PushTSlotDataset(
        slot_data=slot_data["train"],
        split="train",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        action_dir=cfg.action_dir,
        proprio_dir=cfg.proprio_dir,
        state_dir=cfg.get("state_dir", None),
        frameskip=cfg.frameskip,
        seed=cfg.seed,
    )
    
    val_dataset = PushTSlotDataset(
        slot_data=slot_data["val"],
        split="val",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        action_dir=cfg.action_dir,
        proprio_dir=cfg.proprio_dir,
        state_dir=cfg.get("state_dir", None),
        frameskip=cfg.frameskip,
        seed=cfg.seed,
    )
    
    rnd_gen = torch.Generator().manual_seed(cfg.seed)

    train_len = len(train_dataset)
    val_len = len(val_dataset)
    logging.info(f"Train: {train_len}, Val: {val_len}")

    # If datasets are empty, return simple empty iterables to avoid DataLoader errors
    if train_len == 0 and val_len == 0:
        from types import SimpleNamespace
        return SimpleNamespace(train=[], val=[])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        **({"persistent_workers": True} if cfg.num_workers and cfg.num_workers > 0 else {}),
        **({"prefetch_factor": 2} if cfg.num_workers and cfg.num_workers > 0 else {}),
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    try:
        return spt.data.DataModule(train=train_loader, val=val_loader)
    except Exception:
        # If stable_pretraining.data isn't importable (hydra issues), return a simple fallback
        from types import SimpleNamespace
        return SimpleNamespace(train=train_loader, val=val_loader)


# ============================================================================
# Model Architecture
# ============================================================================
def get_world_model(cfg):
    """
    Build world model: masked slot predictor with action/proprio encoders.
    
    Unlike train_causalwm.py, we don't need the DINO encoder and slot attention
    since we're using pre-extracted slots. However, we create placeholder modules
    to maintain checkpoint compatibility.
    """
    
    def forward(self, batch, stage):
        """
        Forward pass using pre-extracted slot embeddings.
        
        This mirrors the forward in train_causalwm.py but skips encoding.
        """
        proprio_key = "proprio" if "proprio" in batch else None
        
        # Replace NaN values with 0 (occurs at sequence boundaries)
        if proprio_key is not None:
            batch[proprio_key] = torch.nan_to_num(batch[proprio_key], 0.0)
        if "action" in batch:
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        
        # Pre-extracted slots are already in the batch as 'pixels_embed'
        # Shape: (B, T, S, D) where D is slot_dim
        pixels_embed = batch["pixels_embed"]  # Pre-extracted slots
        B, T, S, slot_dim = pixels_embed.shape
        
        batch["pixels_embed"] = pixels_embed
        
        # Encode action and proprio (still need to train these)
        embedding = pixels_embed
        
        if proprio_key is not None:
            proprio = batch[proprio_key].float()  # (B, T, proprio_dim)
            proprio_embed = self.model.proprio_encoder(proprio)  # (B, T, proprio_embed_dim)
            batch["proprio_embed"] = proprio_embed
            
            # Tile proprio across slots
            proprio_tiled = proprio_embed.unsqueeze(2)
            embedding = torch.cat([embedding, proprio_tiled], dim=2)
        if "action" in batch:
            action = batch["action"].float()  # (B, T, action_dim * frameskip)
            action_embed = self.model.action_encoder(action)  # (B, T, action_embed_dim)
            batch["action_embed"] = action_embed
            
            # Tile action across slots
            action_tiled = action_embed.unsqueeze(2)
            embedding = torch.cat([embedding, action_tiled], dim=2)
        
        
        batch["embed"] = embedding  # (B, T, S, D_total)
        
        # Use history to predict next states
        history_embed = embedding[:, :cfg.dinowm.history_size, :, :]  # (B, history_size, S+2, 64)
        

        # Request mask information for selective loss
        pred_output = self.model.predict(history_embed)
        slot_num = pixels_embed.shape[2]
        # Get config for Hungarian matching
        use_hungarian = cfg.get("use_hungarian_matching", False)
        hungarian_cost_type = cfg.get("hungarian_cost_type", "mse")
        
        if len(pred_output[1]) > 0:  # mask_indices available
            pred_embedding, mask_indices = pred_output
            target_embedding = embedding[:, cfg.dinowm.history_size:cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]
            
            pred_history = pred_embedding[:, :cfg.dinowm.history_size, :, :]
            pred_future = pred_embedding[:, cfg.dinowm.history_size:cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]
            
            # Loss on masked slots in history (no Hungarian matching for history - slots are aligned)
            gt_history = history_embed
            loss_masked_history = F.mse_loss(
                pred_history[:, :, mask_indices,  :],          #  action/proprio slots already excluded
                gt_history[:, :, mask_indices,  :].detach()    #  action/proprio slots already excluded when selecting mask_indices
            )

            if use_hungarian:
                # Hungarian matching for future slots
                hungarian_result = hungarian_matching_loss_AP(
                    pred=pred_future[:, :, :slot_num, :],
                    target=target_embedding[:, :, :slot_num, :].detach(),
                    cost_type=hungarian_cost_type,
                    reduction="mean",
                )
                loss_future = hungarian_result["pixels_loss"]
                batch["loss_future"] = loss_future
                batch["loss_masked_history"] = loss_masked_history
                batch["loss"] = loss_masked_history + loss_future
                
                if proprio_key is not None: # actually it is future proprio loss
                    loss_proprio = F.mse_loss(
                        pred_future[:, :, slot_num:slot_num+1, :],
                        target_embedding[:, :,  slot_num:slot_num+1, :].detach()
                    )
                    batch['proprio_loss'] = loss_proprio
                    batch["loss"] += loss_proprio
                
                # Log direct MSE for comparison (no gradient - for monitoring only)
                with torch.no_grad():
                    direct_mse_future  = F.mse_loss(pred_future[:, :, :slot_num, :], target_embedding[:, :, :slot_num, :].detach()) # exclude action/proprio slots
                    batch["direct_mse_future_loss"] = direct_mse_future
                    
            else:
                # Original direct MSE loss
                loss_future = F.mse_loss(pred_future[:, :, :slot_num, :], target_embedding[:, :, :slot_num, :].detach()) # exclude action/proprio slots
                
                batch["loss_masked_history"] = loss_masked_history
                batch["loss_future"] = loss_future
                batch["loss"] = loss_masked_history + loss_future
                
                # Add proprio loss if available
                if proprio_key is not None:
                    loss_proprio = F.mse_loss(
                        pred_future[:, :, slot_num:slot_num+1, :],
                        target_embedding[:, :,  slot_num:slot_num+1, :].detach()
                    )
                    batch["proprio_loss"] = loss_proprio
                    batch["loss"] = batch["loss"] + loss_proprio
        else :
            pred_embedding = pred_output[0]
            # pred_future = pred_embedding[:, cfg.dinowm.history_size : cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]       # (B, num_pred, S, 64)
            # target_embedding = batch["embed"][:, cfg.dinowm.history_size : cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]  # (B, num_pred, S, 64)
            # loss_future = F.mse_loss(pred_future[:, :, :slot_num, :], target_embedding[:, :, :slot_num, :].detach()) # exclude action/proprio slots
            # batch["loss"] = loss_future 
            # if proprio_key is not None:
            #     loss_proprio = F.mse_loss(
            #         pred_embedding[:, :, slot_num:slot_num+1, :],
            #         embedding[:, :,  slot_num:slot_num+1, :].detach()
            #     )
            #     batch["loss"] += loss_proprio
            #     batch["loss_proprio"] = loss_proprio
            
            
            pred_history = pred_embedding[:, :cfg.dinowm.history_size, :, :]
            pred_future = pred_embedding[:, cfg.dinowm.history_size:cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]
            target_embedding = embedding[:, cfg.dinowm.history_size:cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]
            
            if use_hungarian:
                # Hungarian matching for future slots
                hungarian_result = hungarian_matching_loss_AP(
                    pred=pred_future[:, :, :slot_num, :],
                    target=target_embedding[:, :, :slot_num, :].detach(),
                    cost_type=hungarian_cost_type,
                    reduction="mean",
                )
                loss_future = hungarian_result["pixels_loss"]
                batch["loss_future"] = loss_future
                batch["loss"] = loss_future
                
                if proprio_key is not None: # actually it is future proprio loss
                    loss_proprio = F.mse_loss(
                        pred_future[:, :, slot_num:slot_num+1, :],
                        target_embedding[:, :,  slot_num:slot_num+1, :].detach()
                    )
                    batch['proprio_loss'] = loss_proprio
                    batch["loss"] += loss_proprio
                
                # Log direct MSE for comparison (no gradient - for monitoring only)
                with torch.no_grad():
                    direct_mse_future  = F.mse_loss(pred_future[:, :, :slot_num, :], target_embedding[:, :, :slot_num, :].detach()) # exclude action/proprio slots
                    batch["direct_mse_future_loss"] = direct_mse_future
            else:
                loss_future = F.mse_loss(pred_future[:, :, :slot_num, :], target_embedding[:, :, :slot_num, :].detach()) # exclude action/proprio slots
                batch["loss_future"] = loss_future
                batch["loss"] = loss_future
                
                # Add proprio loss if available
                if proprio_key is not None:
                    loss_proprio = F.mse_loss(
                        pred_future[:, :, slot_num:slot_num+1, :],
                        target_embedding[:, :,  slot_num:slot_num+1, :].detach()
                    )
                    batch["proprio_loss"] = loss_proprio
                    batch["loss"] = batch["loss"] + loss_proprio
        # Flatten predictions for RankMe: (B, T, S, D) or (B, num_pred, S, D) -> (B*T, S*D) or (B*num_pred, S*D)
        if isinstance(pred_output, tuple) and len(pred_output) > 0:
            B, T, S, D = pred_output[0].shape
            pred_flat = pred_output[0].reshape(B * T, S * D)
        else:
            B, num_pred, S, D = pred_embedding.shape
            pred_flat = pred_embedding.reshape(B * num_pred, S * D)
        batch["predictor_embed"] = pred_flat

        
        # Log losses
        prefix = "train/" if self.training else "val/"
        losses_dict = {f"{prefix}{k}": v.detach() for k, v in batch.items() if "loss" in k}
        self.log_dict(losses_dict, on_step=True, sync_dist=True)

        # Lightweight tensor diagnostics for debugging numeric issues / overfitting.
        # Compute finiteness and simple stats for key tensors (per-batch).
        try:
            stats = {}
            def _add_stats(name, tensor):
                if tensor is None:
                    return
                t = tensor.detach()
                if not isinstance(t, torch.Tensor):
                    return
                finite_mask = torch.isfinite(t)
                nan_count = int((~finite_mask).sum().item())
                # reduce on CPU to avoid GPU sync surprises
                t_cpu = torch.nan_to_num(t).float().cpu()
                if t_cpu.numel() > 0:
                    stats[f"{prefix}{name}_nan_count"] = nan_count
                    stats[f"{prefix}{name}_abs_max"] = float(t_cpu.abs().max().item())
                    stats[f"{prefix}{name}_mean"] = float(t_cpu.mean().item())
                else:
                    stats[f"{prefix}{name}_nan_count"] = nan_count
                    stats[f"{prefix}{name}_abs_max"] = 0.0
                    stats[f"{prefix}{name}_mean"] = 0.0

            # Key tensors to inspect
            _add_stats("pixels_embed", pixels_embed if 'pixels_embed' in locals() else batch.get('pixels_embed', None))
            # pred_embedding/ pred_output may be present
            try:
                _add_stats("pred_embedding", pred_embedding if 'pred_embedding' in locals() else (pred_output[0] if isinstance(pred_output, tuple) else None))
            except Exception:
                pass
            _add_stats("target_embedding", target_embedding if 'target_embedding' in locals() else None)
            _add_stats("action", batch.get("action", None))
            _add_stats("proprio", batch.get(proprio_key, None) if proprio_key is not None else None)

            if len(stats) > 0:
                self.log_dict(stats, on_step=True, sync_dist=True)
        except Exception:
            # Don't let diagnostics break training
            logging.exception("Tensor diagnostics failed")
        
        return batch
    
    # Build the videosaur model to get encoder, slot_attention, initializer
    # These will be frozen and serve as placeholders for checkpoint compatibility
    model = models.build(cfg.model, cfg.dummy_optimizer, None, None)
    encoder = model.encoder
    slot_attention = model.processor
    initializer = model.initializer
    embedding_dim = cfg.videosaur.SLOT_DIM 

    # num_patches = (cfg.image_size // cfg.patch_size) ** 2
    num_patches = cfg.videosaur.NUM_SLOTS
    logging.info(f"Patches: {num_patches}, Embedding dim: {embedding_dim}")

    # Build masked slot predictor (V-JEPA style)
    predictor = MaskedSlot_AP_Predictor(
        num_slots=num_patches + 2,  # number of slots + action_node + proprio_node
        slot_dim=embedding_dim,  # 64 or higher if action/proprio included
        history_frames=cfg.dinowm.history_size,  # T: history length
        pred_frames=cfg.dinowm.num_preds,  # number of future frames to predict
        num_masked_slots=cfg.get("num_masked_slots", 2),  # M: number of slots to mask
        seed=cfg.seed,  # for reproducible masking
        depth=cfg.predictor.get("depth", 6),
        heads=cfg.predictor.get("heads", 16),
        dim_head=cfg.predictor.get("dim_head", 64),
        mlp_dim=cfg.predictor.get("mlp_dim", 2048),
        dropout=cfg.predictor.get("dropout", 0.1),
        future_action_conditioning=cfg.predictor.get("future_action_conditioning", False)
    )
    
    # Build action and proprioception encoders (will be trained)
    effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
    action_encoder = swm.wm.dinowm.Embedder(in_chans=effective_act_dim, emb_dim=cfg.videosaur.SLOT_DIM)
    proprio_encoder = swm.wm.dinowm.Embedder(in_chans=cfg.dinowm.proprio_dim, emb_dim=cfg.videosaur.SLOT_DIM)

    logging.info(f"Action dim: {effective_act_dim}, Proprio dim: {cfg.dinowm.proprio_dim}")

    # Assemble world model
    world_model = CausalWM_AP(
        encoder=spt.backbone.EvalOnly(encoder),
        slot_attention=spt.backbone.EvalOnly(slot_attention),
        initializer=spt.backbone.EvalOnly(initializer),
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
    )
    
    # Wrap in spt.Module with separate optimizers for each trainable component
    def add_opt(module_name, lr):
        return {"modules": str(module_name), "optimizer": {"type": "AdamW", "lr": lr}}
    
    world_model = spt.Module(
        model=world_model,
        forward=forward,
        optim={
            "predictor_opt": add_opt("model.predictor", cfg.predictor_lr),
            "proprio_opt": add_opt("model.proprio_encoder", cfg.proprio_encoder_lr),
            "action_opt": add_opt("model.action_encoder", cfg.action_encoder_lr),
        },
    )
    
    return world_model


# ============================================================================
# Training Setup
# ============================================================================
def setup_pl_logger(cfg):
    """Setup WandB logger for PyTorch Lightning."""
    if not cfg.wandb.enable:
        return None
    
    wandb_run_id = cfg.wandb.get("run_id", None)
    wandb_logger = WandbLogger(
        name="cjepa_ap_slot",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        resume="allow" if wandb_run_id else None,
        id=wandb_run_id,
        log_model=False,
    )
    
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    return wandb_logger


class ModelObjectCallBack(Callback):
    """Callback to save model object after each epoch (same as train_causalwm.py)."""
    
    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)
        
        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                output_path = self.dirpath / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
                # Try to save the underlying model object (preferred) or fallback
                # to saving the LightningModule if necessary.
                try:
                    # If our training wrapper stores the original module under
                    # `module` we can save its `model` attribute which is the
                    # original world-model object.
                    candidate = getattr(pl_module, "module", None)
                    if candidate is not None and hasattr(candidate, "model"):
                        # Prefer saving state_dict to avoid pickling local wrappers
                        try:
                            torch.save(candidate.model.state_dict(), output_path)
                        except Exception:
                            # Fallback to saving the full object if state_dict unavailable
                            torch.save(candidate.model, output_path)
                    else:
                        try:
                            torch.save(pl_module.state_dict(), output_path)
                        except Exception:
                            torch.save(pl_module, output_path)
                    logging.info(f"Saved world model object to {output_path}")
                except Exception:
                    logging.exception("Failed to save world model object")
            
            # Additionally, save at final epoch (also attempt state_dict first)
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                final_path = self.dirpath / f"{self.filename}_object.ckpt"
                try:
                    candidate = getattr(pl_module, "module", None)
                    if candidate is not None and hasattr(candidate, "model"):
                        try:
                            torch.save(candidate.model.state_dict(), final_path)
                        except Exception:
                            torch.save(candidate.model, final_path)
                    else:
                        try:
                            torch.save(pl_module.state_dict(), final_path)
                        except Exception:
                            torch.save(pl_module, final_path)
                    logging.info(f"Saved final world model object to {final_path}")
                except Exception:
                    logging.exception("Failed to save final world model object")

            # Log epoch-level metrics (if any)
            try:
                metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
                logging.info(f"Epoch {trainer.current_epoch+1} metrics: {metrics}")
            except Exception:
                pass


# ============================================================================
# Main Entry Point
# ============================================================================
@hydra.main(version_base=None, config_path="../../configs", config_name="config_train_causal_pusht_slot")
def run(cfg=None):
    """Run training of predictor using pre-extracted slot representations.

    If `cfg` is None (e.g. when invoked via the compatibility wrapper),
    run a minimal smoke test that loads a single batch and validates
    the data pipeline without launching the full trainer.
    """

    if cfg is None:
        # Minimal config for smoke testing
        cfg = OmegaConf.create({
            "cache_dir": "checkpoints",
            "wandb": {"project": "local", "entity": None},
            "output_model_name": "local_smoke",
            "trainer": {"max_epochs": 1},
            "seed": 42,
            "embedding_dir": "checkpoints/train01_slots.pkl",
            "action_dir": "checkpoints/local_action_meta.pkl",
            "proprio_dir": "checkpoints/local_proprio_meta.pkl",
            "state_dir": "checkpoints/local_state_meta.pkl",
            "dinowm": {"history_size": 1, "num_preds": 1},
            "batch_size": 1,
            "num_workers": 0,
            "frameskip": 1,
        })

        # Load data and print one batch to verify shapes
        data = get_data(cfg)
        train_loader = getattr(data, "train", data)
        try:
            batch = next(iter(train_loader))
        except Exception:
            logging.info("Smoke test: no training samples available (dataset length=0). Nothing to run.")
            return

        logging.info("Smoke test: loaded one batch keys: %s", list(batch.keys()))
        for k, v in batch.items():
            try:
                logging.info(f" - {k}: {getattr(v, 'shape', type(v))}")
            except Exception:
                logging.info(f" - {k}: <uninspectable>")
        return

    # Setup cache directory
    cache_dir = Path(swm.data.utils.get_cache_dir() if cfg.cache_dir is None else cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    wandb_logger = setup_pl_logger(cfg)

    # Load data
    data = get_data(cfg)

    # Build world model
    world_model = get_world_model(cfg)
    
    # Setup callbacks
    dump_object_callback = ModelObjectCallBack(
        dirpath=cache_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )
    
    callbacks = [dump_object_callback]
    

    
    # Setup trainer
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=wandb_logger,
        enable_checkpointing=True,
    )
    
    # Run training
    # Allow overriding the checkpoint path to resume from a Lightning checkpoint
    # Useful when resuming from a previous run's `events` checkpoint file.
    ckpt_override = cfg.get("resume_ckpt_path", None)
    if ckpt_override:
        ckpt_path = str(ckpt_override)
    else:
        ckpt_path = str(cache_dir / f"{cfg.output_model_name}_weights.ckpt")

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data,
        ckpt_path=ckpt_path,
        seed=cfg.seed,
    )
    manager()


if __name__ == "__main__":
    run()
