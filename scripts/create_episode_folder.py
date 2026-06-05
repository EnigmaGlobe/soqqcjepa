#!/usr/bin/env python
"""Create a single episode folder by filtering CSVs on `episode_id` and copying the main video."""
from pathlib import Path
import argparse
import pandas as pd
import shutil
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
    p.add_argument('--ep', type=int, required=True)
    p.add_argument('--out_root', required=True)
    p.add_argument('--video', required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    csvs = list(data_dir.glob('*.csv'))
    if not csvs:
        print('No CSVs found in', data_dir)
        return

    for csv in csvs:
        df = pd.read_csv(csv)
        if 'episode_id' not in df.columns:
            print('CSV has no episode_id:', csv)
            continue
        ep_df = df[df['episode_id'] == args.ep]
        if ep_df.empty:
            print(f'No rows for episode {args.ep} in {csv.name}')
            continue
        folder = out_root / f'episode_{args.ep}'
        folder.mkdir(parents=True, exist_ok=True)
        ep_df.to_csv(folder / csv.name, index=False)
        print('Wrote', folder / csv.name)

    # compute start/end frame across CSVs if possible
    all_start = None
    all_end = None
    for csv in csvs:
        df = pd.read_csv(csv, usecols=lambda c: c in ['episode_id','frame_count','step_index','realtime_since_start'])
        ep_df = df[df.get('episode_id') == args.ep]
        if ep_df.empty:
            continue
        if 'frame_count' in ep_df.columns:
            s = int(ep_df['frame_count'].min())
            e = int(ep_df['frame_count'].max())
        elif 'step_index' in ep_df.columns:
            s = int(ep_df['step_index'].min())
            e = int(ep_df['step_index'].max())
        else:
            s = None
            e = None

        if s is not None:
            all_start = s if all_start is None else min(all_start, s)
        if e is not None:
            all_end = e if all_end is None else max(all_end, e)

    v = Path(args.video)
    out_folder = out_root / f'episode_{args.ep}'
    out_video = out_folder / f'episode_{args.ep}.mp4'
    if v.exists() and all_start is not None and all_end is not None and all_end >= all_start:
        try:
            print(f'Extracting frames {all_start}-{all_end} from {v} to {out_video}')
            extract_video_segment(v, out_video, all_start, all_end)
            print('Extracted video segment to', out_video)
        except Exception as e:
            print('Failed to extract segment, copying full video:', e)
            shutil.copy2(v, out_video)
    else:
        if v.exists():
            shutil.copy2(v, out_video)
            print('Copied full video to', out_video)
        else:
            print('Video not found:', v)


if __name__ == '__main__':
    main()
