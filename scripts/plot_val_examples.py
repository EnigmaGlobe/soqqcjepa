import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def find_arrays(d):
    # search dict for candidate prediction/target arrays
    preds = None
    targets = None

    if isinstance(d, dict):
        for k, v in d.items():
            lk = k.lower()
            if lk in ("predictions", "preds", "pred", "y_hat", "yhat") and preds is None:
                preds = v
            if lk in ("targets", "target", "gt", "y") and targets is None:
                targets = v
            if preds is None or targets is None:
                p, t = find_arrays(v)
                if preds is None and p is not None:
                    preds = p
                if targets is None and t is not None:
                    targets = t
    elif isinstance(d, list):
        for item in d:
            p, t = find_arrays(item)
            if preds is None and p is not None:
                preds = p
            if targets is None and t is not None:
                targets = t
    return preds, targets


def ensure_np(x):
    if x is None:
        return None
    arr = np.array(x)
    return arr


def plot_example(pred, targ, outpath, example_idx=0, max_dims=8):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    # pred, targ: (T, S, D) or (B, T, S, D) or (B, T, D)
    if pred.ndim == 4:
        pred = pred[example_idx]
    if targ.ndim == 4:
        targ = targ[example_idx]

    # collapse slots if present
    if pred.ndim == 3:
        # (T, S, D) -> (T, S*D)
        T, S, D = pred.shape
        pred_flat = pred.reshape(T, S * D)
    else:
        pred_flat = pred
    if targ.ndim == 3:
        T2, S2, D2 = targ.shape
        targ_flat = targ.reshape(T2, S2 * D2)
    else:
        targ_flat = targ

    T = min(pred_flat.shape[0], targ_flat.shape[0])
    pred_flat = pred_flat[:T]
    targ_flat = targ_flat[:T]

    dims = min(pred_flat.shape[1], max_dims)

    plt.figure(figsize=(12, 6))
    for i in range(dims):
        plt.plot(pred_flat[:, i], alpha=0.6, label=f'pred_dim{i}' if i == 0 else None)
        plt.plot(targ_flat[:, i], '--', alpha=0.6, label=f'tgt_dim{i}' if i == 0 else None)

    err = np.mean((pred_flat - targ_flat) ** 2, axis=1)
    err = (err - err.min()) / (err.max() - err.min() + 1e-9)
    plt.plot(err * np.max(np.abs(targ_flat)) , color='k', linewidth=1, label='norm_mse')

    plt.legend(loc='upper right', fontsize='small')
    plt.title(f'Example {example_idx} — first {dims} dims')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--outdir', default='outputs/val_examples')
    p.add_argument('--per-stage', type=int, default=3)
    p.add_argument('--max-dims', type=int, default=8)
    args = p.parse_args()

    if args.input.endswith('.npz'):
        import numpy as _np
        data_np = _np.load(args.input)
        preds = ensure_np(data_np['preds'])
        targs = ensure_np(data_np['targs'])
    else:
        with open(args.input, 'r') as f:
            data = json.load(f)

        preds, targs = find_arrays(data)
        preds = ensure_np(preds)
        targs = ensure_np(targs)

    if preds is None or targs is None:
        print('Could not find predictions or targets in', args.input)
        return

    # If data appears per-stage (dict with stages), handle that
    if isinstance(preds, dict):
        for stage, pvals in preds.items():
            tv = targs.get(stage, None) if isinstance(targs, dict) else targs
            p_arr = ensure_np(pvals)
            t_arr = ensure_np(tv)
            if p_arr is None or t_arr is None:
                continue
            for i in range(min(args.per_stage, p_arr.shape[0])):
                out = os.path.join(args.outdir, f'stage_{stage}_example_{i}.png')
                plot_example(p_arr, t_arr, out, example_idx=i, max_dims=args.max_dims)
    else:
        # assume first dim is batch
        for i in range(min(args.per_stage, preds.shape[0])):
            out = os.path.join(args.outdir, f'example_{i}.png')
            plot_example(preds, targs, out, example_idx=i, max_dims=args.max_dims)


if __name__ == '__main__':
    main()
