"""Generate per-frame projected slot embeddings for local videos.
Produces a pickle with structure {"train": {key: ndarray(T,S,D)}, "val": {}}

Usage:
  python scripts/generate_slots_proj.py --video-dir testdata/train01 --out checkpoints/train01_slots_proj2.pkl --device cuda
"""
import argparse
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn.functional as F


def process_video(path: Path, device='cpu', S=4, D=128, batch_size=1024):
    import imageio

    # First pass: count frames to allocate output
    r = imageio.get_reader(str(path))
    T = 0
    for _ in r:
        T += 1
    r.close()
    if T == 0:
        return np.zeros((0, S, D), dtype=np.float32)

    # Use a temporary memmap to store results while streaming frames
    import tempfile, os
    tmpf = Path(tempfile.gettempdir()) / f"{path.stem}_slots_tmp.npy"
    if tmpf.exists():
        try:
            tmpf.unlink()
        except Exception:
            pass
    mm = np.memmap(str(tmpf), dtype=np.float32, mode='w+', shape=(T, S, D))

    # Second pass: stream frames and process in batches, writing to memmap
    device_t = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    pool = torch.nn.AdaptiveAvgPool2d((4, 4)).to(device_t)
    rng = np.random.RandomState(0)
    Wproj = None

    r = imageio.get_reader(str(path))
    idx = 0
    buffer = []
    for frame in r:
        buffer.append(np.ascontiguousarray(frame[:, :, ::-1]))
        if len(buffer) >= batch_size:
            arr = np.stack(buffer)
            buffer = []
            # to torch
            video_batch = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
            video_batch = video_batch.to(device_t)
            B = video_batch.shape[0]
            C = video_batch.shape[1]
            # prepare patches for this batch
            H = video_batch.shape[2]; W = video_batch.shape[3]
            h_mid = H // 2; w_mid = W // 2
            patches_b = [
                video_batch[:, :, :h_mid, :w_mid].contiguous(),
                video_batch[:, :, :h_mid, w_mid:].contiguous(),
                video_batch[:, :, h_mid:, :w_mid].contiguous(),
                video_batch[:, :, h_mid:, w_mid:].contiguous(),
            ]
            if Wproj is None:
                proj_in = C * 4 * 4
                Wproj = torch.from_numpy(rng.normal(scale=0.1, size=(proj_in, D)).astype(np.float32)).to(device_t)

            for slot_idx, p in enumerate(patches_b[:S]):
                try:
                    p_res = F.interpolate(p, size=(32, 32), mode='bilinear', align_corners=False)
                except Exception:
                    p_res = p
                pooled = pool(p_res)
                flat = pooled.view(pooled.shape[0], -1)
                proj_out = flat.matmul(Wproj)
                mm[idx:idx+proj_out.shape[0], slot_idx, :] = proj_out.detach().cpu().numpy()
            idx += B
    # leftover buffer
    if buffer:
        arr = np.stack(buffer)
        video_batch = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
        video_batch = video_batch.to(device_t)
        B = video_batch.shape[0]
        C = video_batch.shape[1]
        H = video_batch.shape[2]; W = video_batch.shape[3]
        h_mid = H // 2; w_mid = W // 2
        patches_b = [
            video_batch[:, :, :h_mid, :w_mid].contiguous(),
            video_batch[:, :, :h_mid, w_mid:].contiguous(),
            video_batch[:, :, h_mid:, :w_mid].contiguous(),
            video_batch[:, :, h_mid:, w_mid:].contiguous(),
        ]
        if Wproj is None:
            proj_in = C * 4 * 4
            Wproj = torch.from_numpy(rng.normal(scale=0.1, size=(proj_in, D)).astype(np.float32)).to(device_t)
        for slot_idx, p in enumerate(patches_b[:S]):
            try:
                p_res = F.interpolate(p, size=(32, 32), mode='bilinear', align_corners=False)
            except Exception:
                p_res = p
            pooled = pool(p_res)
            flat = pooled.view(pooled.shape[0], -1)
            proj_out = flat.matmul(Wproj)
            mm[idx:idx+proj_out.shape[0], slot_idx, :] = proj_out.detach().cpu().numpy()
        idx += B

    r.close()
    # load memmap into a normal ndarray (fits in memory for typical slot dims)
    out_arr = np.array(mm)
    try:
        os.remove(str(tmpf))
    except Exception:
        pass
    return out_arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    video_dir = Path(args.video_dir)
    files = sorted([p for p in video_dir.glob('**/*.mp4')])
    if not files:
        raise SystemExit('No mp4 files found')

    out = {'train': {}, 'val': {}}
    for f in files:
        key = f"{f.stem}_pixels.mp4"
        print('Processing', f)
        slots = process_video(f, device=args.device)
        print(' ->', slots.shape)
        out['train'][key] = slots

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as fh:
        pickle.dump(out, fh)
    print('Wrote', out_path)

if __name__ == '__main__':
    main()
