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
    # Prefer torchvision-based reader to avoid torchcodec native dependency
    try:
        from torchvision.io import read_video
        from torchvision import transforms as tvt
        # read_video returns video as Tensor[T, H, W, C]
        video, _, _ = read_video(str(path))
        if video is None or len(video) == 0:
            return np.zeros((0, 0, 0), dtype=np.float32)
        # normalize to float in [0,1] and convert to [T,C,H,W]
        video = video.float() / 255.0
        video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
    except Exception:
        # Fallback to cv2 if torchvision read_video is unavailable
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame[:, :, ::-1])
            cap.release()
            if len(frames) == 0:
                return np.zeros((0, 0, 0), dtype=np.float32)
            video = torch.from_numpy(np.stack(frames)).float() / 255.0
            video = video.permute(0, 3, 1, 2)
        except Exception:
            return np.zeros((0, 0, 0), dtype=np.float32)

    # If the model exposes encoder+processor like Videosaur ObjectCentricModel, use them
    try:
        # resize + normalize similar to extract_videosaur: Resize -> Normalize
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        tfs = tvt.Compose([
            tvt.ConvertImageDtype(torch.float32),
            tvt.Resize((196, 196)),
            tvt.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        video_t = tfs(video) if 'tfs' in locals() else video
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
    T = video.shape[0]
    S = getattr(model, 'NUM_SLOTS', 4)
    D = getattr(model, 'SLOT_DIM', 128)
    return np.zeros((T, S, D), dtype=np.float32)


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
        if hasattr(args, 'videosaur_config') and getattr(args, 'videosaur_config', None) and args.ckpt is not None:
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

        # fallback: build a minimal dummy model if conf not provided or build failed
        if model is None:
            model_cfg = {"model": {"name": "videosaur_dummy"}}
            dummy_optimizer = {"name": "Adam", "lr": 0.001}
            model = models.build(model_cfg, dummy_optimizer, None)

        # If a checkpoint path was passed, try to load it into the model.
        if args.ckpt is not None:
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
                print(f"Warning: failed to load checkpoint {args.ckpt}: {e}; continuing with built model or falling back to dummy slots")

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
    except Exception as e:
        print(f"Warning: videosaur model build failed: {e}; will create dummy zero slots instead.")

    files = sorted([p for p in video_dir.glob('**/*.mp4')])
    if not files:
        raise SystemExit(f"No mp4 files found in {video_dir}")

    out = {"train": {}, "val": {}}

    for p in files:
        key = f"{p.stem}_pixels.mp4"
        print(f"Processing {p} -> key {key}")
        if model is not None:
            slots = extract_slots_for_file(model, p, device=device)
        else:
            # Count frames with cv2 and create zero slots: S=cfg default 4, D=128
            try:
                import cv2
                cap = cv2.VideoCapture(str(p))
                cnt = 0
                while True:
                    ok, _ = cap.read()
                    if not ok:
                        break
                    cnt += 1
                cap.release()
            except Exception:
                cnt = 0
            S = 4
            D = 128
            print(f"Creating dummy slots array with T={cnt}, S={S}, D={D}")
            slots = np.zeros((cnt, S, D), dtype=np.float32)
        out['train'][key] = slots

    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, 'wb') as f:
        pkl.dump(out, f)
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
