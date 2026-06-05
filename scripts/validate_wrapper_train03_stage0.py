import importlib.util
import json
from omegaconf import OmegaConf

# Load the validate module as a library
spec = importlib.util.spec_from_file_location("vm", "scripts/validate_mlagents_counterfactual.py")
vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)

cfg = OmegaConf.load('configs/config_validate_train03_full.yaml')
ckpt = r'C:\Users\infra\\.stable_worldmodel\\local_run_bs128_ep50_weights.ckpt'
device = 'cpu'

world_model = vm.load_world_model(ckpt, cfg, device=device)
val_loader = vm.build_val_loader(cfg)

standard_metrics = vm.standard_validation(world_model, val_loader, cfg, device, max_batches=50)
action_cf = vm.counterfactual_action_probe(world_model, val_loader, cfg, device, max_batches=20)
slot_cf = vm.counterfactual_slot_ablation(world_model, val_loader, cfg, device, max_batches=20)
stage_cf = vm.training_stage_consistency_probe(world_model, val_loader, cfg, device, max_batches=20)

results = {
    'checkpoint': ckpt,
    'standard': standard_metrics,
    'counterfactual_action': action_cf,
    'counterfactual_slot_ablation': slot_cf,
    'training_stage_consistency': stage_cf,
}

with open('checkpoints/validation_results_train03_stage0.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Saved results to checkpoints/validation_results_train03_stage0.json')
