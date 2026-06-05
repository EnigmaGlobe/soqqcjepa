import numpy as np
from pathlib import Path

npz = Path('outputs/val_error_maps/error_mse.npz')
if not npz.exists():
    print('Missing', npz)
    raise SystemExit(1)

data = np.load(npz)
mse_time_slot = data['mse_time_slot']  # (T, S)

# per-slot average over time
per_slot = mse_time_slot.mean(axis=0)
per_time = mse_time_slot.mean(axis=1)

print('per_slot_mse:', per_slot.tolist())
print('\nTop slots by MSE:')
for idx, val in sorted(enumerate(per_slot), key=lambda x: x[1], reverse=True):
    print(f' slot {idx}: {val:.6e}')

print('\nper_time_mse:', per_time.tolist())
print('\nSummary: mean MSE {:.6e}, std {:.6e}'.format(per_slot.mean(), per_slot.std()))
