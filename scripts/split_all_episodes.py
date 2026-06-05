#!/usr/bin/env python
"""Split all episodes found in the data_training CSVs into per-episode folders.

Usage:
  python scripts/split_all_episodes.py --data_dir testdata\1\data_training --video testdata\1\train_01_recording.mp4 --out_dir testdata\1\episodes

This will create a folder per episode `episode_{id}` containing filtered CSVs and an extracted video `episode_{id}.mp4`.
"""
from pathlib import Path
import argparse
import pandas as pd
import shutil
import os
import cv2


def extract_video_segment(video_path, out_path, start_frame, end_frame):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    cur = start_frame
    while cur <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        cur += 1

    writer.release()
    cap.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--max_eps', type=int, default=None, help='Optional max episodes to process')
    p.add_argument('--skip_existing', action='store_true', help='Skip episodes whose output files already exist')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)

    obs_csv = data_dir / 'observations_frame_train_01.csv'
    actions_csv = data_dir / 'actions_frame_train_01.csv'
    if not obs_csv.exists():
        print('observations CSV not found:', obs_csv)
        return

    obs_df = pd.read_csv(obs_csv)
    act_df = pd.read_csv(actions_csv) if actions_csv.exists() else None
    eps = sorted(obs_df['episode_id'].unique())
    if args.max_eps:
        eps = eps[:args.max_eps]

    total = len(eps)
    print(f'Processing {total} episodes')

    for i, ep in enumerate(eps, 1):
        ep_folder = out_root / f'episode_{ep}'
        ep_folder.mkdir(parents=True, exist_ok=True)
        obs_out = ep_folder / obs_csv.name
        act_out = ep_folder / actions_csv.name
        out_video = ep_folder / f'episode_{ep}.mp4'

        if args.skip_existing and obs_out.exists() and out_video.exists() and (act_df is None or act_out.exists()):
            if i % 50 == 0 or i == total:
                print(f'Completed {i}/{total} episodes')
            continue

        # filter and write observations
        ep_obs = obs_df[obs_df['episode_id'] == ep]
        try:
            ep_obs.to_csv(obs_out, index=False)
        except Exception as e:
            print(f'Warning: failed to write observations for episode {ep}: {e}; skipping episode')
            continue

        # filter actions if present
        if act_df is not None:
            ep_act = act_df[act_df['episode_id'] == ep]
            try:
                ep_act.to_csv(act_out, index=False)
            except Exception as e:
                print(f'Warning: failed to write actions for episode {ep}: {e}; continuing')

        # compute frame range
        start_frame = None
        end_frame = None
        if 'frame_count' in ep_obs.columns:
            start_frame = int(ep_obs['frame_count'].min())
            end_frame = int(ep_obs['frame_count'].max())
        elif 'step_index' in ep_obs.columns:
            start_frame = int(ep_obs['step_index'].min())
            end_frame = int(ep_obs['step_index'].max())

        if video.exists() and start_frame is not None and end_frame is not None and end_frame >= start_frame:
            try:
                extract_video_segment(video, out_video, start_frame, end_frame)
            except Exception as e:
                print('Failed to extract for ep', ep, 'error:', e)
                shutil.copy2(video, out_video)
        else:
            if video.exists():
                shutil.copy2(video, out_video)

        if i % 50 == 0 or i == total:
            print(f'Completed {i}/{total} episodes')


if __name__ == '__main__':
    main()
