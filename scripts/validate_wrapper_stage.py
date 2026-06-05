import argparse
import importlib.util
import json
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slots', required=True, help='Path to stage slots pickle')
    parser.add_argument('--actions', required=True, help='Path to stage actions pickle')
    parser.add_argument('--proprio', required=True, help='Path to stage proprio pickle')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--out', default='checkpoints/validation_results_stage.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_batches', type=int, default=200)
    parser.add_argument('--cf_batches', type=int, default=100)
    args = parser.parse_args()

    # Load validation module
    spec = importlib.util.spec_from_file_location('vm', 'scripts/validate_mlagents_counterfactual.py')
    vm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vm)

    cfg = OmegaConf.load('configs/config_validate_train03_full.yaml')
    # Override embedding/action/proprio paths
    cfg.embedding_dir = args.slots
    cfg.action_dir = args.actions
    cfg.proprio_dir = args.proprio
    # Avoid loading stale state pickles that don't match this stage
    cfg.state_dir = None

    world_model = vm.load_world_model(args.checkpoint, cfg, device=args.device)
    val_loader = vm.build_val_loader(cfg)

    standard = vm.standard_validation(world_model, val_loader, cfg, args.device, max_batches=args.max_batches)
    action_cf = vm.counterfactual_action_probe(world_model, val_loader, cfg, args.device, max_batches=args.cf_batches)
    slot_cf = vm.counterfactual_slot_ablation(world_model, val_loader, cfg, args.device, max_batches=args.cf_batches)
    stage_cf = vm.training_stage_consistency_probe(world_model, val_loader, cfg, args.device, max_batches=args.cf_batches)

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
