import csv
import sys
from statistics import mean, stdev

path = sys.argv[1] if len(sys.argv)>1 else 'checkpoints/local_run_bs128_ep50_metrics.csv'
rows = []
with open(path, newline='') as fh:
    r = csv.DictReader(fh)
    for row in r:
        rows.append(row)
if not rows:
    print('No rows found')
    sys.exit(1)

# helper to get float with fallback
def f(v):
    try:
        return float(v)
    except:
        return float('nan')

steps = [int(r['step']) for r in rows]
epochs = [float(r.get('epoch', 0)) for r in rows]

metrics = ['train/loss','train/loss_future','train/loss_masked_history','train/pixels_embed_abs_max','train/pixels_embed_mean','train/pixels_embed_nan_count','train/proprio_loss','train/action_abs_max']

out = {}
for m in metrics:
    vals = [f(r.get(m,'')) for r in rows]
    vals_no_nan = [v for v in vals if not (v!=v)]
    out[m] = {
        'first': vals[0],
        'last': vals[-1],
        'min': min(vals_no_nan) if vals_no_nan else float('nan'),
        'max': max(vals_no_nan) if vals_no_nan else float('nan'),
        'mean': mean(vals_no_nan) if vals_no_nan else float('nan'),
        'stdev': stdev(vals_no_nan) if len(vals_no_nan)>1 else 0.0,
    }

# find best epoch by train/loss min
losses = [f(r.get('train/loss','')) for r in rows]
best_idx = min(range(len(losses)), key=lambda i: losses[i])
best = rows[best_idx]

print('Summary for', path)
print('Steps:', len(steps), 'First step', steps[0], 'Last step', steps[-1])
print('Best step (min train/loss):', best_idx, 'step', best.get('step'), 'epoch', best.get('epoch'), 'train/loss=', best.get('train/loss'))
print('\nKey metrics:')
for m in metrics:
    v = out[m]
    print(f"- {m}: first={v['first']:.6g} last={v['last']:.6g} min={v['min']:.6g} max={v['max']:.6g} mean={v['mean']:.6g} stdev={v['stdev']:.6g}")

# NaN checks: any _nan_count > 0
nan_tags = [k for k in rows[0].keys() if k.endswith('_nan_count')]
print('\nNaN counts present:')
for tag in nan_tags:
    vals = [f(r.get(tag,'')) for r in rows]
    ssum = sum(v for v in vals if not (v!=v))
    if ssum>0:
        print(f"- {tag}: sum={ssum} (non-zero present)")
    else:
        print(f"- {tag}: all zero")

# show loss change
first_loss = out['train/loss']['first']
last_loss = out['train/loss']['last']
if not (first_loss!=first_loss) and not (last_loss!=last_loss):
    pct = (last_loss-first_loss)/first_loss*100 if first_loss!=0 else float('nan')
    print(f"\nTrain loss: first={first_loss:.6g}, last={last_loss:.6g}, change={pct:.2f}%")

print('\nEnd of summary')
