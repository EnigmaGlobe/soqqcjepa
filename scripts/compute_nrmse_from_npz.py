import numpy as np
from pathlib import Path

npz = Path('outputs/tmp_val_examples.npz')
if not npz.exists():
    print('Missing', npz)
    raise SystemExit(1)

data = np.load(npz)
preds = data['preds']  # (B, T, S, D)
targs = data['targs']

assert preds.shape == targs.shape

diff2 = (preds - targs) ** 2
# per-slot MSE averaged over batch,time,feature
mse_per_slot = diff2.mean(axis=(0,1,3))
rmse_per_slot = np.sqrt(mse_per_slot)

# target std per slot (over batch,time,feature)
std_per_slot = targs.reshape(-1, targs.shape[2], targs.shape[3]).std(axis=0).mean(axis=1)

# handle zero std
std_per_slot = np.where(std_per_slot < 1e-9, 1e-9, std_per_slot)

nrmse_per_slot = rmse_per_slot / std_per_slot

print('RMSE per slot:', rmse_per_slot.tolist())
print('STD per slot:', std_per_slot.tolist())
print('NRMSE per slot:', nrmse_per_slot.tolist())
print('\nOverall RMSE {:.6e}, Overall STD {:.6e}, Overall NRMSE {:.6e}'.format(
    np.sqrt(diff2.mean()), targs.std(), np.sqrt(diff2.mean()) / (targs.std() + 1e-9)
))
