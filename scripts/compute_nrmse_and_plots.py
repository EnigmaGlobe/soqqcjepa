import os
import glob
import csv
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = Path('checkpoints/val_by_stage.csv')
OUT_CSV = Path('checkpoints/val_by_stage_nrmse.csv')
OUT_PNG = Path('outputs/val_by_stage_nrmse.png')
os.makedirs('outputs', exist_ok=True)

# load val_by_stage.csv
rows = []
with CSV_PATH.open() as fh:
    r = csv.DictReader(fh)
    for row in r:
        rows.append(row)

results = []
for row in rows:
    stage = row['stage']
    mse = float(row.get('val_future_mse', 'nan'))
    # find matching slots pickle in checkpoints/validation_staged
    pattern = f"checkpoints/validation_staged/*{stage}*slots.pkl"
    matches = glob.glob(pattern)
    if not matches:
        # try other pattern
        matches = glob.glob(f"checkpoints/validation_staged/*{stage}*slots*.pkl")
    if not matches:
        print('No slots pickle for', stage)
        std_all = float('nan')
        var_all = float('nan')
        mean_sq = float('nan')
    else:
        p = matches[0]
        with open(p,'rb') as f:
            d = pickle.load(f)
        # d may be {'train':..., 'val':...} or direct mapping
        if isinstance(d, dict) and ('train' in d or 'val' in d):
            slot_map = d.get('val', d.get('train', {}))
        else:
            slot_map = d
        # collect all values
        vals = []
        for k, arr in slot_map.items():
            a = np.asarray(arr)
            if a.ndim == 3:
                # (T,S,D)
                vals.append(a.reshape(-1))
            elif a.ndim == 4 and a.shape[0]==1:
                vals.append(a.squeeze(0).reshape(-1))
            else:
                vals.append(a.reshape(-1))
        if len(vals)==0:
            std_all = float('nan')
            var_all = float('nan')
            mean_sq = float('nan')
        else:
            allvals = np.concatenate(vals, axis=0)
            std_all = float(np.std(allvals))
            var_all = float(np.var(allvals))
            mean_sq = float(np.mean(allvals**2))
    rmse = np.sqrt(mse)
    nrmse = rmse / std_all if (not np.isnan(std_all) and std_all>0) else float('nan')
    results.append({'stage':stage, 'val_future_mse':mse, 'rmse':rmse, 'std_target':std_all, 'nrmse':nrmse, 'var_target':var_all, 'mean_sq':mean_sq})

# write CSV
with OUT_CSV.open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
    w.writeheader()
    for r in results:
        w.writerow(r)

# plot bar chart: RMSE vs std_target and baseline RMSE (sqrt(var)=std)
stages = [r['stage'] for r in results]
rmse = [r['rmse'] for r in results]
stds = [r['std_target'] for r in results]
vars_ = [r['var_target'] for r in results]

x = np.arange(len(stages))
width = 0.25
plt.figure(figsize=(10,5))
plt.bar(x - width, rmse, width, label='RMSE')
plt.bar(x, stds, width, label='STD(target)')
plt.bar(x + width, [np.sqrt(v) if not np.isnan(v) else np.nan for v in vars_], width, label='sqrt(Var)=STD')
plt.xticks(x, stages)
plt.ylabel('Value')
plt.title('RMSE vs Target STD by Stage')
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PNG)
print('Wrote', OUT_CSV, 'and', OUT_PNG)
