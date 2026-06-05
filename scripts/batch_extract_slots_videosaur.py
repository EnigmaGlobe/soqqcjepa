#!/usr/bin/env python
"""Batch extract slot embeddings from a folder of videos using Videosaur.

Produces a single combined pickle with structure {'train':{basename: slots}, 'val':{}}
so it's directly consumable by `PushTSlotDataset`.

Usage:
  python scripts/batch_extract_slots_videosaur.py --weight /path/to/ckpt --input_dir testdata/train01 --output testdata/train01/combined_slots.pkl --input_size 196 --frame_skip 1 --videosaur_config src/third_party/videosaur/configs/videosaur/pusht_dinov2_hf.yml
"""
import argparse
import os
import glob
import pickle
import torch
import numpy as np
from pathlib import Path

from src.third_party.videosaur.videosaur import configuration, models
from scripts.extract_slots_videosaur_opencv import read_video_opencv, resize_frames, normalize_frames, build_video_tensor


def load_model(weight, videosaur_config=None):
    conf = None
    if videosaur_config:
        try:
            conf = configuration.load_config(videosaur_config)
        except Exception:
            conf = None

    model = None
    if conf is not None:
        model = models.build(conf.model, conf.optimizer)

    ckpt = torch.load(weight, map_location=torch.device('cpu'))
    if model is None:
        if conf is not None:
            model = models.build(conf.model, conf.optimizer)
        else:
            raise RuntimeError("Provide a compatible videosaur config to build model for the checkpoint")

    model.load_state_dict(ckpt["state_dict"])  # may raise
    model.eval()
    return model, conf


def process_video(model, conf, video_path, input_size=196, frame_skip=1):
    frames = read_video_opencv(video_path, frame_skip=frame_skip)
    frames = resize_frames(frames, input_size)
    frames = normalize_frames(frames)
    video_tensor = build_video_tensor(frames)

    with torch.no_grad():
        inputs = {"video": video_tensor}
        outputs = model(inputs)
        processor_out = outputs.get('processor', {})
        if 'state' in processor_out:
            slots = processor_out['state'][0].cpu().numpy()
        elif 'all_slot_states' in processor_out:
            slots = processor_out['all_slot_states'][0].cpu().numpy()
        else:
            raise RuntimeError("Could not find slot states in model outputs")

    return slots


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weight', required=True)
    p.add_argument('--input_dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--input_size', type=int, default=196)
    p.add_argument('--frame_skip', type=int, default=1)
    p.add_argument('--videosaur_config', default=None)
    args = p.parse_args()

    videos = sorted(glob.glob(os.path.join(args.input_dir, '*.mp4')))
    if len(videos) == 0:
        print('No mp4 files found in', args.input_dir)
        return

    print(f'Loading Videosaur model from {args.weight} ...')
    model, conf = load_model(args.weight, args.videosaur_config)
    combined = {'train': {}, 'val': {}}

    for v in videos:
        name = os.path.basename(v)
        print('Processing', name)
        try:
            slots = process_video(model, conf, v, input_size=args.input_size, frame_skip=args.frame_skip)
            combined['train'][name] = slots
            print(f' -> extracted slots shape {slots.shape}')
        except Exception as e:
            print('Failed to process', v, e)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, 'wb') as f:
        pickle.dump(combined, f)

    print('Wrote combined slots to', args.output)


if __name__ == '__main__':
    main()
