import argparse
import json
import torch
from omegaconf import OmegaConf
import sys
from pathlib import Path

# import the original module to reuse functions
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.validate_mlagents_counterfactual as vmc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="validation_results_safe.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--cf_batches", type=int, default=50)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"[1/5] Loading model from {args.checkpoint}")
    world_model = vmc.load_world_model(args.checkpoint, cfg, device=device)

    print("[2/5] Building validation loader")
    val_loader = vmc.build_val_loader(cfg)
    print(f"      Validation samples: {len(val_loader.dataset)}")

    print("[3/5] Running standard validation")
    standard_metrics = vmc.standard_validation(
        world_model, val_loader, cfg, device=device, max_batches=args.max_batches
    )
    print(f"      {standard_metrics}")

    print("[4/5] Counterfactual Probe A — Action Intervention")
    action_cf = vmc.counterfactual_action_probe(
        world_model, val_loader, cfg, device=device, max_batches=args.cf_batches
    )
    print(f"      zero_action_div={action_cf['action_zero_divergence_mse']:.6f}  "
          f"rand_action_div={action_cf['action_random_divergence_mse']:.6f}")

    print("[5/5] Counterfactual Probe B — Slot Ablation")
    slot_cf = vmc.counterfactual_slot_ablation(
        world_model, val_loader, cfg, device=device, max_batches=args.cf_batches
    )
    print(f"      most_causal_slot={slot_cf['most_causal_slot_index']}  "
          f"least_causal_slot={slot_cf['least_causal_slot_index']}")
    print(f"      per-slot importance: {[f'{x:.4f}' for x in slot_cf['slot_ablation_importance']]}")

    print("[Bonus] Training-Stage Consistency Probe")
    stage_cf = vmc.training_stage_consistency_probe(
        world_model, val_loader, cfg, device=device, max_batches=args.cf_batches
    )
    print(f"      action_history_swap_div={stage_cf['action_history_swap_divergence_mse']:.6f}")

    # Safe serialization: avoid crashing on OmegaConf eval interpolations
    try:
        cfg_container = OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        cfg_container = f"OMEGACONF_SERIALIZATION_ERROR: {str(e)}"

    results = {
        "config": cfg_container,
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
