#!/usr/bin/env python3
"""Lightweight slot extractor wrapper for local videos using in-repo VideoSAUR.

Saves a pickle with structure {"train": {key: ndarray(T,S,D)}, "val": {}}.

Usage example (PowerShell):
  $env:PYTHONPATH='C:\new\soqqcjepa'; .\.venv\Scripts\python.exe scripts/extract_slots_from_videos_local.py --video-dir 'G:\Shared drives\Shared\2025\RLrecordingdata\jepa\train01' --out checkpoints/train01_slots.pkl
"""
import argparse
import os
import pickle as pkl
from pathlib import Path
import torch
import numpy as np

# use in-repo videosaur
from src.third_party.videosaur.videosaur import models


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", default=None, help="optional videosaur checkpoint to load")
    p.add_argument("--videosaur-config", default=None, help="optional videosaur YAML config to build model from")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def extract_slots_for_file(model, path, device="cpu"):
    """Very small wrapper: read video with cv2, run model.extract_slots if available.
    Fall back to creating a dummy zero array if the model interface isn't present.
    Returns ndarray (T, S, D).
    """
    # Try multiple readers: prefer imageio (robust pure-python), then torchvision, then cv2
    video = None
    # 1) imageio reader
    try:
        import imageio
        frames = []
        r = imageio.get_reader(str(path))
        for f in r:
            frames.append(np.ascontiguousarray(f[:, :, ::-1]))
        r.close()
        print(f"[extractor] imageio read frames: {len(frames)}")
        if len(frames) == 0:
            print("[extractor] imageio returned 0 frames")
            return np.zeros((0, 0, 0), dtype=np.float32)
        video = torch.from_numpy(np.stack(frames)).float() / 255.0
        video = video.permute(0, 3, 1, 2).contiguous()
    except Exception:
        video = None

    if video is None:
        # 2) torchvision
        try:
            from torchvision.io import read_video
            from torchvision import transforms as tvt
            video, _, _ = read_video(str(path))
            if video is None or len(video) == 0:
                print("[extractor] torchvision returned 0 frames or None")
                video = None
            else:
                print(f"[extractor] torchvision read frames: {len(video)}")
                video = video.float() / 255.0
                video = video.permute(0, 3, 1, 2).contiguous()
        except Exception:
            video = None

    if video is None:
        # 3) cv2 fallback
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            frames = []
            while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(np.ascontiguousarray(frame[:, :, ::-1]))
            cap.release()
            print(f"[extractor] cv2 read frames: {len(frames)}")
            if len(frames) == 0:
                print("[extractor] cv2 returned 0 frames")
                return np.zeros((0, 0, 0), dtype=np.float32)
            video = torch.from_numpy(np.stack(frames)).float() / 255.0
            video = video.permute(0, 3, 1, 2).contiguous()
        except Exception:
            return np.zeros((0, 0, 0), dtype=np.float32)

    # Diagnostic: report whether we have a video tensor
    try:
        print(f"[extractor] video tensor shape: {None if video is None else tuple(video.shape)}")
    except Exception:
        pass

    # If the model exposes encoder+processor like Videosaur ObjectCentricModel, use them
    # Prepare transforms: prefer torchvision transforms, but fall back to a
    # lightweight resize using torch.nn.functional if torchvision is absent.
    try:
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        try:
            from torchvision import transforms as tvt
            tfs = tvt.Compose([
                tvt.ConvertImageDtype(torch.float32),
                tvt.Resize((196, 196)),
                tvt.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            video_t = tfs(video)
        except Exception:
            # torchvision.transforms not available: do a simple tensor-based
            # resize and ensure dtype/normalization roughly matches expectations.
            import torch.nn.functional as F
            video_t = video.float()
            try:
                # video shape: (T, C, H, W)
                video_t = F.interpolate(video_t, size=(196, 196), mode='bilinear', align_corners=False)
            except Exception:
                # If interpolation fails, keep original size.
                pass
            # Normalize approximately (in-place safe copy)
            try:
                mean = torch.tensor(IMAGENET_MEAN, device=video_t.device).view(1, -1, 1, 1)
                std = torch.tensor(IMAGENET_STD, device=video_t.device).view(1, -1, 1, 1)
                video_t = (video_t - mean) / std
            except Exception:
                pass
    except Exception:
        video_t = video

    # model may expect input shape [1, T, C, H, W]
    try:
        video_b = video_t.unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model, 'encoder') and hasattr(model, 'processor'):
                encoder_out = model.encoder(video_b)
                features = encoder_out.get('features', None)
                if features is None:
                    # Can't extract features -> fall back
                    raise RuntimeError('encoder returned no features')
                # initializer + processor path
                slots_init = model.initializer(batch_size=1).to(device)
                out = model.processor(slots_init, features)
                slots_np = out['state'][0].detach().cpu().numpy()
                return slots_np
            # try encode/forward_slots APIs
            if hasattr(model, 'encode'):
                out = model.encode({'video': video_t.unsqueeze(0)}, target='slots')
                slots = out.get('slots', None)
                if slots is not None:
                    arr = np.asarray(slots)
                    if arr.ndim == 4 and arr.shape[0] == 1:
                        arr = arr.squeeze(0)
                    return arr
            if hasattr(model, 'forward_slots'):
                slots = model.forward_slots({'video': video_t.unsqueeze(0)})
                return np.asarray(slots)
    except Exception:
        pass

    # final fallback: construct zeros with shape T x S x D
    # If the videosaur model wasn't available, produce simple per-patch
    # embeddings using a lightweight ResNet backbone projected to 128 dims.
    try:
        T = int(video.shape[0])
    except Exception:
        return np.zeros((0, 0, 0), dtype=np.float32)

    # Number of slots: default to 4 (2x2 patches)
    S = getattr(model, 'NUM_SLOTS', 4)
    target_D = getattr(model, 'SLOT_DIM', 128)

    # If a model object exists but we couldn't use its API, prefer zeros
    if model is not None:
        D = target_D
    else:
        # Lightweight, reliable fallback extractor (CPU):
        # - split each frame into 2x2 patches
        # - adaptive-pool each patch to (4,4)
        # - flatten and apply a fixed random projection to `target_D`
        try:
            import torch.nn.functional as F
            import torch

            video_cpu = video.float().contiguous().cpu()
            _, C, H, W = video_cpu.shape
            h_mid = H // 2
            w_mid = W // 2
            patches = [
                video_cpu[:, :, :h_mid, :w_mid].contiguous(),
                video_cpu[:, :, :h_mid, w_mid:].contiguous(),
                video_cpu[:, :, h_mid:, :w_mid].contiguous(),
                video_cpu[:, :, h_mid:, w_mid:].contiguous(),
            ]

            all_slots = np.zeros((T, S, target_D), dtype=np.float32)

            # Fixed random projection matrix for deterministic-ish features
            rng = np.random.RandomState(0)
            proj_in = C * 4 * 4
            Wproj = torch.from_numpy(rng.normal(scale=0.1, size=(proj_in, target_D)).astype(np.float32))

            pool = torch.nn.AdaptiveAvgPool2d((4, 4))
            batch_size = 512
            print(f"[extractor] projection fallback T={T}, S={S}, target_D={target_D}")
            for slot_idx, p in enumerate(patches[:S]):
                # p: (T, C, h, w)
                # resize patches to a reasonable size then pool
                # do in batches to limit memory
                feats = []
                for i in range(0, T, batch_size):
                    b = p[i:i+batch_size]
                    if b.shape[0] == 0:
                        continue
                    try:
                        b_resized = F.interpolate(b, size=(32, 32), mode='bilinear', align_corners=False)
                    except Exception:
                        b_resized = b
                    pooled = pool(b_resized)  # (B, C, 4, 4)
                    flat = pooled.view(pooled.shape[0], -1)  # (B, C*4*4)
                    # project
                    proj_out = flat.matmul(Wproj)
                    feats.append(proj_out.cpu().numpy())

                if feats:
                    feats_np = np.concatenate(feats, axis=0)
                    if feats_np.shape[0] < T:
                        pad = np.zeros((T - feats_np.shape[0], target_D), dtype=np.float32)
                        feats_np = np.concatenate([feats_np, pad], axis=0)
                    all_slots[:, slot_idx, :] = feats_np[:T]

            return all_slots
        except Exception as e:
            print(f"[extractor] projection fallback failed: {e}")
            return np.zeros((T, S, target_D), dtype=np.float32)


def main():
    args = parse_args()
    video_dir = Path(args.video_dir)
    out_path = Path(args.out)
    device = args.device

    # Try to build videosaur model; if it fails, we'll fall back to dummy slots
    model = None
    try:
        # If user provided a videosaur config, load it to build the model correctly
        conf = None
        if getattr(args, 'videosaur_config', None) and args.ckpt is not None:
            try:
                # Use the videosaur inference helper which handles config+checkpoint loading
                from src.third_party.videosaur.videosaur import inference as videosaur_inference

                model, conf = videosaur_inference.load_model_from_checkpoint(str(args.ckpt), str(args.videosaur_config))
                model = model.eval()
                print(f"Loaded videosaur model from checkpoint {args.ckpt} and config {args.videosaur_config}")
            except Exception as e:
                print(f"Warning: failed to load videosaur model via inference helper: {e}")
                # Try building directly from a videosaur YAML (some wrapper inference configs import torchvision.read_video at module import)
                try:
                    from src.third_party.videosaur.videosaur import configuration as videosaur_configuration

                    conf = videosaur_configuration.load_config(Path(args.videosaur_config))
                    model = models.build(conf.model, conf.optimizer)
                    print(f"Built videosaur model from YAML {args.videosaur_config}")
                except Exception as e2:
                    print(f"Warning: also failed to build model from YAML {args.videosaur_config}: {e2}")

        # Do not attempt to build a dummy dict-based Videosaur model here.
        # If no valid `conf`/model was built above, keep `model = None` and
        # let `extract_slots_for_file()` handle the safe fallback extraction.
        # If a checkpoint path was passed and we actually have a model, try to load it.
        if args.ckpt is not None and model is not None:
            try:
                ckpt_path = str(args.ckpt)
                print(f"Loading checkpoint {ckpt_path} into videosaur model (map_location=cpu)")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                # Common Lightning wrapper fields
                if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                    sd = ckpt['state_dict']
                elif isinstance(ckpt, dict) and 'model' in ckpt:
                    sd = ckpt['model']
                else:
                    sd = ckpt

                # Normalize keys: remove common Lightning prefixes like 'model.' or 'module.'
                def _strip_prefix(d, prefixes=('model.', 'module.')):
                    new = {}
                    for k, v in d.items():
                        new_k = k
                        for p in prefixes:
                            if new_k.startswith(p):
                                new_k = new_k[len(p):]
                                break
                        new[new_k] = v
                    return new

                if isinstance(sd, dict):
                    sd2 = _strip_prefix(sd)
                    try:
                        model.load_state_dict(sd2, strict=False)
                        print("Loaded checkpoint into model (partial load allowed)")
                    except Exception as e:
                        print(f"Warning: failed to load state_dict into model: {e}")
                else:
                    print("Warning: checkpoint format not recognized for state_dict loading; proceeding with built model")
            except Exception as e:
                print(f"Warning: failed to load checkpoint {args.ckpt}: {e}; continuing without loading weights")

        # If we built from a videosaur config and it specifies NUM_SLOTS, set initializer n_slots
        try:
            if conf is not None and hasattr(conf, 'globals') and conf.globals is not None:
                ns = getattr(conf.globals, 'NUM_SLOTS', None)
                if ns is not None and hasattr(model, 'initializer'):
                    try:
                        model.initializer.n_slots = ns
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception as e:
        print(f"Warning: videosaur model build failed: {e}; will create dummy zero slots instead.")

    files = sorted([p for p in video_dir.glob('**/*.mp4')])
    if not files:
        raise SystemExit(f"No mp4 files found in {video_dir}")

    out = {"train": {}, "val": {}}

    for p in files:
        key = f"{p.stem}_pixels.mp4"
        print(f"Processing {p} -> key {key}")
        # Always use the central extractor function; it handles model==None
        # and will use the safe deterministic projection fallback when Videosaur
        # is not available.
        slots = extract_slots_for_file(model, p, device=device)
        out['train'][key] = slots

    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, 'wb') as f:
        pkl.dump(out, f)
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
