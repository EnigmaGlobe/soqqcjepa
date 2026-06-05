import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='NPZ with preds and targs')
    p.add_argument('--outdir', default='outputs')
    args = p.parse_args()

    data = np.load(args.input)
    preds = data['preds']  # shape (B, T, S, D)
    targs = data['targs']  # shape (B, T, S, D)

    assert preds.shape == targs.shape, f'shape mismatch {preds.shape} vs {targs.shape}'

    # Mean squared error per time and slot (average over batch and feature dim)
    diff2 = (preds - targs) ** 2
    mse_time_slot = diff2.mean(axis=(0, 3))  # shape (T, S)

    # transpose to (S, T) for plotting with slots on y
    mse_slot_time = mse_time_slot.T

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(mse_slot_time, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label='MSE')
    plt.xlabel('prediction timestep')
    plt.ylabel('slot index')
    plt.title('Per-slot × time MSE')
    plt.tight_layout()
    plt.savefig(outdir / 'error_heatmap_slot_time.png')
    plt.close()

    # Per-slot aggregate
    per_slot = mse_slot_time.mean(axis=1)
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(per_slot)), per_slot)
    plt.xlabel('slot index')
    plt.ylabel('MSE')
    plt.title('Per-slot MSE (avg over time)')
    plt.tight_layout()
    plt.savefig(outdir / 'error_per_slot.png')
    plt.close()

    # Per-time aggregate
    per_time = mse_slot_time.mean(axis=0)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(per_time)), per_time, marker='o')
    plt.xlabel('prediction timestep')
    plt.ylabel('MSE')
    plt.title('Per-time MSE (avg over slots)')
    plt.tight_layout()
    plt.savefig(outdir / 'error_per_time.png')
    plt.close()

    # Save numpy
    np.savez(outdir / 'error_mse.npz', mse_time_slot=mse_time_slot, mse_slot_time=mse_slot_time)
    print('Saved heatmap and aggregates to', outdir)


if __name__ == '__main__':
    main()
