import argparse
import os
import pickle
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from src.third_party.videosaur.videosaur import configuration, models


def read_video_opencv(path, frame_skip=1):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            # skip broken frame
            continue
        try:
            # handle frames with unexpected channels
            if frame.shape[-1] == 4:
                # BGRA -> BGR by dropping alpha
                frame = frame[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Warning: skipping frame due to cvtColor error: {e}")
            continue
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames read from video: {path}")
    frames = np.asarray(frames)
    if frame_skip > 1:
        frames = frames[::frame_skip]
    return frames  # shape (T, H, W, C)


def resize_frames(frames, size):
    h, w = size, size
    out = []
    for f in frames:
        out.append(cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR))
    return np.stack(out, axis=0)


def normalize_frames(frames):
    # Use MOVI_DEFAULT normalization from videosaur (mean=0.5,std=0.5) to match inference path
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    frames = frames.astype(np.float32) / 255.0
    frames = (frames - mean[None, None, None, :]) / std[None, None, None, :]
    return frames


def build_video_tensor(frames_np):
    # frames_np: (T, H, W, C) RGB float32 already normalized
    frames = np.transpose(frames_np, (0, 3, 1, 2))  # (T, C, H, W)
    tensor = torch.from_numpy(frames)
    # model expects (B, T, C, H, W)
    tensor = tensor.unsqueeze(0)
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True, help="Videosaur checkpoint (.ckpt)")
    parser.add_argument("--video", required=True, help="Path to input MP4")
    parser.add_argument("--save_path", default="checkpoints/test_videosaur_slots_opencv.pkl")
    parser.add_argument("--input_size", type=int, default=196)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--videosaur_config", default=None, help="optional videosaur YAML to infer n_slots")
    args = parser.parse_args()

    print(f"Loading model from {args.weight}")
    conf = None
    if args.videosaur_config:
        # The provided config may be an "inference" yaml that references the real model config.
        try:
            # Try to load as inference-style config first to find referenced model_config
            inf_conf = OmegaConf.load(args.videosaur_config)
            model_cfg_path = None
            if isinstance(inf_conf, dict) or hasattr(inf_conf, 'get'):
                model_cfg_path = inf_conf.get('model_config') if inf_conf.get('model_config') else None
            if model_cfg_path:
                # resolve relative paths commonly used in the repo
                candidates = [model_cfg_path, os.path.join('src', model_cfg_path), os.path.join('src', 'third_party', model_cfg_path)]
                for c in candidates:
                    if os.path.exists(c):
                        conf = configuration.load_config(c)
                        break
                if conf is None:
                    # fallback: try direct load if path exists
                    try:
                        conf = configuration.load_config(model_cfg_path)
                    except Exception:
                        print(f"Could not resolve model config path: {model_cfg_path}")
            else:
                # maybe user passed the model config directly
                try:
                    conf = configuration.load_config(args.videosaur_config)
                except Exception as e:
                    print(f"Failed to load videosaur config as model config: {e}")
        except Exception as e:
            # as last resort, try to load directly with configuration.load_config
            try:
                conf = configuration.load_config(args.videosaur_config)
            except Exception as e2:
                print(f"Failed to load videosaur config: {e}; {e2}")

    model = None
    try:
        # build model from checkpoint config if possible
        if conf is not None:
            model = models.build(conf.model, conf.optimizer)
        else:
            # fallback: build with minimal args (models.build requires model_config and optimizer_config)
            # Attempt to load checkpoint and infer state dict into a default build using config in checkpoint
            # If that fails, load state dict into an empty model build isn't possible and will raise.
            model = None
    except Exception as e:
        print(f"Warning: building model with config failed: {e}")
        model = None

    ckpt = torch.load(args.weight, map_location=torch.device('cpu'))

    if model is None:
        # Try to import from videosaur.models.build by loading checkpoint first: the checkpoint alone isn't sufficient.
        # As a last resort, try to load a state_dict into a cloned model by importing model class dynamically from checkpoint.
        # Simpler: reuse the inference.load_model_from_checkpoint behavior: try to load via configuration if provided.
        if conf is not None:
            model = models.build(conf.model, conf.optimizer)
        else:
            raise RuntimeError("Could not build Videosaur model: provide --videosaur_config matching checkpoint")

    model.load_state_dict(ckpt["state_dict"])  # may raise
    model.eval()

    if conf is not None:
        try:
            if hasattr(conf, 'globals') and hasattr(conf.globals, 'NUM_SLOTS'):
                model.initializer.n_slots = int(conf.globals.NUM_SLOTS)
        except Exception:
            pass

    print(f"Reading video {args.video}")
    frames = read_video_opencv(args.video, frame_skip=args.frame_skip)
    print(f"Read {len(frames)} frames")
    frames = resize_frames(frames, args.input_size)
    frames = normalize_frames(frames)
    video_tensor = build_video_tensor(frames)  # (1, T, C, H, W)

    print(f"Video tensor shape: {video_tensor.shape}")

    with torch.no_grad():
        inputs = {"video": video_tensor}
        outputs = model(inputs)
        # processor state lives in outputs['processor']['state'] with shape [B, T, n_slots, dim]
        processor_out = outputs.get('processor', {})
        if 'state' in processor_out:
            slots = processor_out['state'][0].cpu().numpy()
        elif 'all_slot_states' in processor_out:
            # fallback
            slots = processor_out['all_slot_states'][0].cpu().numpy()
        else:
            raise RuntimeError("Could not find slot states in model outputs")

    print(f"Extracted slots array shape: {slots.shape}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Save in the same dict format expected by training code
    basename = os.path.basename(args.video)
    slots_dict = {"train": {basename: slots}, "val": {}}
    with open(args.save_path, 'wb') as f:
        pickle.dump(slots_dict, f)

    print(f"Saved slots pickle to {args.save_path}")


if __name__ == '__main__':
    main()
