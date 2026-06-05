"""
validate_mlagents_counterfactual.py
====================================
Final validation + counterfactual probing for ML-Agents C-JEPA models.

This script runs three evaluations on held-out (new-seed) validation data:

1. Standard Validation
   Future-prediction MSE in slot space (same as training val loop).

2. Counterfactual Probe A — Action Intervention
   Replace true actions with zero / random / mean actions and measure how
   much the predicted future slots diverge.  Large divergence means the
   model understands that actions causally drive the world.

3. Counterfactual Probe B — Slot Ablation (Causal Attribution)
   Zero-out each slot individually in the history and predict the future.
   The slot whose removal hurts prediction the most is the most causally
   important one (for ML-Agents this should be the agent slot).

Usage
-----
python scripts/validate_mlagents_counterfactual.py \
    --config configs/config_train_causal_pusht_slot.yaml \
    --checkpoint checkpoints/local_run_bs64_ep30_epoch_30_object.ckpt \
    --output validation_results.json \
    --max_batches 100

Dataset setup (see README section below)
----------------------------------------
- Your ~2h training video stays in the **train** split.
- New-seed videos go into the **val** split exclusively.
- Extract slots / actions / proprio from the new-seed videos into the same
  pickle format your training script consumed:
      slot_pkl      = {"train": {vid: slots_arr, ...}, "val": {vid: slots_arr, ...}}
      action_pkl    = {"train": {vid: actions_arr, ...}, "val": {vid: actions_arr, ...}}
      proprio_pkl   = {"train": {vid: proprio_arr, ...}, "val": {vid: proprio_arr, ...}}
- If you want to diagnose generalisation across training stages, you can
  further sub-divide the val pickle into e.g. val-early / val-mid / val-late
  and run this script three times with different `--embedding_dir` paths.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from einops import repeat

# ---------------------------------------------------------------------------
# Repo path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cjepa_predictor import MaskedSlotPredictor
from src.world_models.dinowm_causal import CausalWM
from src.custom_codes.custom_dataset import PushTSlotDataset
from src.third_party.videosaur.videosaur import models
import stable_pretraining as spt
import stable_worldmodel as swm


# =============================================================================
# Model loading helpers
# =============================================================================

def build_world_model(cfg: OmegaConf):
    """Construct a fresh CausalWM from Hydra config (encoder frozen)."""
    # Build the videosaur backbone so we get compatible checkpoint keys.
    # The encoder / slot_attention / initializer are frozen placeholders here
    # because we run on pre-extracted slots.
    model = models.build(cfg.model, cfg.dummy_optimizer, None, None)
    encoder = model.encoder
    slot_attention = model.processor
    initializer = model.initializer

    slot_dim = cfg.videosaur.SLOT_DIM
    num_slots = cfg.videosaur.NUM_SLOTS

    # Predictor expects to operate on slot-shaped embeddings. Some trained
    # checkpoints incorporate action/proprio by projecting them into the slot
    # dimension and adding them to slot embeddings (keeping predictor input
    # dim == slot_dim). To be compatible with such checkpoints we set the
    # predictor's `slot_dim` to the base `slot_dim` rather than the concat
    # embedding size; at runtime we will inject action/proprio into the slot
    # vectors (additive) when the encoders output slot-sized vectors.
    predictor = MaskedSlotPredictor(
        num_slots=num_slots,
        slot_dim=slot_dim,
        history_frames=cfg.dinowm.history_size,
        pred_frames=cfg.dinowm.num_preds,
        num_masked_slots=cfg.get("num_masked_slots", 2),
        seed=cfg.seed,
        depth=cfg.predictor.get("depth", 6),
        heads=cfg.predictor.get("heads", 16),
        dim_head=cfg.predictor.get("dim_head", 64),
        mlp_dim=cfg.predictor.get("mlp_dim", 2048),
        dropout=cfg.predictor.get("dropout", 0.1),
    )

    effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
    # Build encoders to project actions/proprio into the slot dimension so
    # they can be injected (added) into slot embeddings. This matches the
    # training-time setup for checkpoints that contain action/proprio
    # encoders producing slot-sized outputs.
    action_encoder = swm.wm.dinowm.Embedder(
        in_chans=effective_act_dim, emb_dim=slot_dim
    )
    proprio_encoder = swm.wm.dinowm.Embedder(
        in_chans=cfg.dinowm.proprio_dim, emb_dim=slot_dim
    )

    world_model = CausalWM(
        encoder=spt.backbone.EvalOnly(encoder),
        slot_attention=spt.backbone.EvalOnly(slot_attention),
        initializer=spt.backbone.EvalOnly(initializer),
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
    )
    return world_model


def load_world_model(ckpt_path: str, cfg: OmegaConf, device: str = "cuda"):
    """Build + load checkpoint, handling multiple save formats."""
    world_model = build_world_model(cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract raw state dict from possible wrapper types
    if hasattr(ckpt, "state_dict"):
        state_dict = ckpt.state_dict()
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Strip common prefixes so keys match bare CausalWM
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model.model."):
            cleaned[k[12:]] = v          # LightningModule -> spt.Module -> CausalWM
        elif k.startswith("model."):
            cleaned[k[6:]] = v           # spt.Module -> CausalWM
        else:
            cleaned[k] = v

    missing, unexpected = world_model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    world_model.eval().to(device)
    return world_model


# =============================================================================
# Data
# =============================================================================

def build_val_loader(cfg: OmegaConf):
    """Instantiate validation DataLoader from pre-extracted pickles."""
    with open(cfg.embedding_dir, "rb") as f:
        slot_data = pickle.load(f)

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

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    return val_loader


# =============================================================================
# Evaluation kernels
# =============================================================================

def _build_embedding(world_model, slots, actions, proprios, num_slots):
    """Replicate training forward: encode action/proprio and tile across slots."""
    embedding = slots  # (B, T, S, D_slot)

    # If the action/proprio encoders project to the same dimension as slots,
    # the original training may have added them into the slot vectors (so the
    # predictor input dim remained `slot_dim`). In that case we add the
    # projected encodings to the slot embeddings. Otherwise we fall back to
    # concatenation (older configs where predictor expects larger dim).
    slot_dim = embedding.shape[-1]

    if proprios is not None:
        proprios = torch.nan_to_num(proprios, 0.0)
        p_emb = world_model.proprio_encoder(proprios.float())
        # p_emb: (B, T, p_dim)
        if p_emb.shape[-1] == slot_dim:
            p_tiled = repeat(p_emb.unsqueeze(2), "b t 1 d -> b t p d", p=num_slots)
            embedding = embedding + p_tiled
        else:
            p_tiled = repeat(p_emb.unsqueeze(2), "b t 1 d -> b t p d", p=num_slots)
            embedding = torch.cat([embedding, p_tiled], dim=-1)

    if actions is not None:
        actions = torch.nan_to_num(actions, 0.0)
        a_emb = world_model.action_encoder(actions.float())
        if a_emb.shape[-1] == slot_dim:
            a_tiled = repeat(a_emb.unsqueeze(2), "b t 1 d -> b t p d", p=num_slots)
            embedding = embedding + a_tiled
        else:
            a_tiled = repeat(a_emb.unsqueeze(2), "b t 1 d -> b t p d", p=num_slots)
            embedding = torch.cat([embedding, a_tiled], dim=-1)

    return embedding


@torch.no_grad()
def standard_validation(world_model, val_loader, cfg, device, max_batches=None):
    """Future-prediction MSE using predictor.inference (no masking)."""
    slot_dim = cfg.videosaur.SLOT_DIM
    losses = []

    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        slots = batch["pixels_embed"].to(device)
        actions = batch["action"].to(device) if "action" in batch else None
        proprios = batch["proprio"].to(device) if "proprio" in batch else None
        B, T, S, D = slots.shape

        emb = _build_embedding(world_model, slots, actions, proprios, S)
        hist = emb[:, : cfg.dinowm.history_size, :, :]
        pred = world_model.predict(hist, use_inference_function=True)[..., :slot_dim]

        tgt = emb[:, cfg.dinowm.history_size : cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :][
            ..., :slot_dim
        ]
        losses.append(F.mse_loss(pred, tgt).item())

    return {"val_future_mse": float(np.mean(losses)), "val_future_std": float(np.std(losses))}


@torch.no_grad()
def counterfactual_action_probe(world_model, val_loader, cfg, device, max_batches=50):
    """
    Probe 1: Action Intervention
    ----------------------------
    Compare predictions under TRUE actions vs ZERO actions vs RANDOM actions.
    Metrics:
        - action_zero_divergence_mse   : ||pred_true - pred_zero||^2
        - action_random_divergence_mse : ||pred_true - pred_random||^2
        - per_slot_action_sensitivity  : which slots move most under zero action?
    """
    slot_dim = cfg.videosaur.SLOT_DIM
    zero_deltas, rand_deltas = [], []
    slot_sensitivities = []

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        slots = batch["pixels_embed"].to(device)
        actions = batch["action"].to(device) if "action" in batch else None
        proprios = batch["proprio"].to(device) if "proprio" in batch else None
        B, T, S, D = slots.shape

        emb_true = _build_embedding(world_model, slots, actions, proprios, S)

        # Counterfactual 1: zero action
        actions_zero = torch.zeros_like(actions) if actions is not None else None
        emb_zero = _build_embedding(world_model, slots, actions_zero, proprios, S)

        # Counterfactual 2: random action (Gaussian noise at same scale)
        if actions is not None:
            act_std = actions.std(dim=(0, 1), keepdim=True).clamp_min(1e-3)
            actions_rand = torch.randn_like(actions) * act_std
        else:
            actions_rand = None
        emb_rand = _build_embedding(world_model, slots, actions_rand, proprios, S)

        # Predict futures
        hist_true = emb_true[:, : cfg.dinowm.history_size, :, :]
        hist_zero = emb_zero[:, : cfg.dinowm.history_size, :, :]
        hist_rand = emb_rand[:, : cfg.dinowm.history_size, :, :]

        pred_true = world_model.predict(hist_true, use_inference_function=True)[..., :slot_dim]
        pred_zero = world_model.predict(hist_zero, use_inference_function=True)[..., :slot_dim]
        pred_rand = world_model.predict(hist_rand, use_inference_function=True)[..., :slot_dim]

        zero_deltas.append(F.mse_loss(pred_true, pred_zero).item())
        rand_deltas.append(F.mse_loss(pred_true, pred_rand).item())

        # Per-slot sensitivity to zero action (mean over B,T,D)
        per_slot = ((pred_true - pred_zero) ** 2).mean(dim=(0, 1, 3))  # (S,)
        slot_sensitivities.append(per_slot.cpu().numpy())

    return {
        "action_zero_divergence_mse": float(np.mean(zero_deltas)),
        "action_random_divergence_mse": float(np.mean(rand_deltas)),
        "per_slot_action_sensitivity": np.stack(slot_sensitivities).mean(axis=0).tolist(),
    }


@torch.no_grad()
def counterfactual_slot_ablation(world_model, val_loader, cfg, device, max_batches=50):
    """
    Probe 2: Slot Ablation
    ----------------------
    For each slot, zero it out in the history and predict future.
    The slot whose ablation increases MSE the most is the most causally
    important slot (analogous to CLEVRER's "what if object X vanished?").
    """
    slot_dim = cfg.videosaur.SLOT_DIM
    importances = []

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        slots = batch["pixels_embed"].to(device)
        actions = batch["action"].to(device) if "action" in batch else None
        proprios = batch["proprio"].to(device) if "proprio" in batch else None
        B, T, S, D = slots.shape

        emb = _build_embedding(world_model, slots, actions, proprios, S)
        hist = emb[:, : cfg.dinowm.history_size, :, :]
        pred_base = world_model.predict(hist, use_inference_function=True)[..., :slot_dim]

        batch_imp = []
        for s in range(S):
            hist_abl = hist.clone()
            hist_abl[:, :, s, :] = 0.0
            pred_abl = world_model.predict(hist_abl, use_inference_function=True)[..., :slot_dim]
            batch_imp.append(F.mse_loss(pred_abl, pred_base).item())
        importances.append(batch_imp)

    avg_imp = np.mean(importances, axis=0)
    return {
        "slot_ablation_importance": avg_imp.tolist(),
        "most_causal_slot_index": int(np.argmax(avg_imp)),
        "least_causal_slot_index": int(np.argmin(avg_imp)),
    }


@torch.no_grad()
def training_stage_consistency_probe(world_model, val_loader, cfg, device, max_batches=50):
    """
    Probe 3: Training-Stage Consistency (unique to your non-stationary setup)
    -------------------------------------------------------------------------
    Since your video captures policy *training*, the same visual state can
    appear with different action distributions at different times.
    This probe measures whether the model has learned to condition its
    predictions on the action history (i.e. policy) rather than just physics.

    Metric: action_history_swap_mse
        - Take two random windows from the SAME validation video.
        - Replace the action sequence in window A with actions from window B
          (keeping the visual history of A).
        - Measure prediction divergence.
        - If the model ignores the action swap (low divergence), it has not
          disentangled policy from world dynamics.
    """
    slot_dim = cfg.videosaur.SLOT_DIM
    swaps = []

    # We need pairs of windows; simplest way: iterate two at a time
    it = iter(val_loader)
    for _ in range(max_batches):
        try:
            batch_a = next(it)
            batch_b = next(it)
        except StopIteration:
            break

        slots_a = batch_a["pixels_embed"].to(device)
        actions_a = batch_a["action"].to(device) if "action" in batch_a else None
        proprios_a = batch_a["proprio"].to(device) if "proprio" in batch_a else None

        actions_b = batch_b["action"].to(device) if "action" in batch_b else None

        _, T, S, _ = slots_a.shape

        emb_true = _build_embedding(world_model, slots_a, actions_a, proprios_a, S)
        emb_swap = _build_embedding(world_model, slots_a, actions_b, proprios_a, S)

        hist_true = emb_true[:, : cfg.dinowm.history_size, :, :]
        hist_swap = emb_swap[:, : cfg.dinowm.history_size, :, :]

        pred_true = world_model.predict(hist_true, use_inference_function=True)[..., :slot_dim]
        pred_swap = world_model.predict(hist_swap, use_inference_function=True)[..., :slot_dim]

        swaps.append(F.mse_loss(pred_true, pred_swap).item())

    return {
        "action_history_swap_divergence_mse": float(np.mean(swaps)),
        "num_pairs": len(swaps),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ML-Agents C-JEPA validation + counterfactual probes")
    parser.add_argument("--config", required=True, help="Hydra config YAML (e.g. configs/config_train_causal_pusht_slot.yaml)")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.ckpt)")
    parser.add_argument("--output", default="validation_results.json", help="JSON output path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_batches", type=int, default=None, help="Cap batches for quick smoke tests")
    parser.add_argument("--cf_batches", type=int, default=50, help="Batches to use for counterfactual probes")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"[1/5] Loading model from {args.checkpoint}")
    world_model = load_world_model(args.checkpoint, cfg, device=device)

    print("[2/5] Building validation loader")
    val_loader = build_val_loader(cfg)
    print(f"      Validation samples: {len(val_loader.dataset)}")

    print("[3/5] Running standard validation")
    standard_metrics = standard_validation(
        world_model, val_loader, cfg, device=device, max_batches=args.max_batches
    )
    print(f"      {standard_metrics}")

    print("[4/5] Counterfactual Probe A — Action Intervention")
    action_cf = counterfactual_action_probe(
        world_model, val_loader, cfg, device=device, max_batches=args.cf_batches
    )
    print(f"      zero_action_div={action_cf['action_zero_divergence_mse']:.6f}  "
          f"rand_action_div={action_cf['action_random_divergence_mse']:.6f}")

    print("[5/5] Counterfactual Probe B — Slot Ablation")
    slot_cf = counterfactual_slot_ablation(
        world_model, val_loader, cfg, device=device, max_batches=args.cf_batches
    )
    print(f"      most_causal_slot={slot_cf['most_causal_slot_index']}  "
          f"least_causal_slot={slot_cf['least_causal_slot_index']}")
    print(f"      per-slot importance: {[f'{x:.4f}' for x in slot_cf['slot_ablation_importance']]}")

    print("[Bonus] Training-Stage Consistency Probe")
    stage_cf = training_stage_consistency_probe(
        world_model, val_loader, cfg, device=device, max_batches=args.cf_batches
    )
    print(f"      action_history_swap_div={stage_cf['action_history_swap_divergence_mse']:.6f}")

    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "checkpoint": args.checkpoint,
        "standard": standard_metrics,
        "counterfactual_action": action_cf,
        "counterfactual_slot_ablation": slot_cf,
        "training_stage_consistency": stage_cf,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main()
