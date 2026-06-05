#!/usr/bin/env python
"""Split videos and CSVs by episode into per-episode folders.

Example usage (try first 2 episodes):
  python scripts/split_video_by_episode.py --input_dir testdata\1 --output_dir testdata\1\episodes --max_episodes 2

Behavior:
- If multiple CSV files found, the script will try to infer an episode id from each CSV filename and copy each pair (csv + matching video) into episode folders.
- If a single CSV file is found, the script will look for an episode id column (episode, episode_id, ep, traj) and split the CSV by episode. If frame/time columns exist, the script will cut the source video into per-episode video segments.
- If no clear episode/time columns are found, the script will create per-episode CSVs and copy the source video into each episode folder so you can verify structure.
"""
from pathlib import Path
import argparse
import re
import shutil
import sys
import csv
import os

try:
	import pandas as pd
except Exception:
	pd = None

import cv2


EPISODE_CANDIDATES = ['episode', 'episode_id', 'ep', 'traj', 'trial', 'episodeIdx', 'episode_idx']
START_TIME_CANDIDATES = ['start_time', 'start_sec', 'start_s', 'start_ms', 'start_frame', 'start_idx']
END_TIME_CANDIDATES = ['end_time', 'end_sec', 'end_s', 'end_ms', 'end_frame', 'end_idx']
SINGLE_FRAME_CANDIDATES = ['frame', 'frame_idx', 'frame_index']


def find_files(input_dir):
	p = Path(input_dir)
	videos = list(p.glob('*.mp4')) + list(p.glob('*.avi'))
	csvs = list(p.glob('*.csv'))
	return videos, csvs


def extract_episode_from_name(name):
	# try simple patterns
	for pat in [r'episode[_-]?(\d+)', r'ep[_-]?(\d+)', r'_(\d{1,3})', r'-([0-9]{1,3})']:
		m = re.search(pat, name)
		if m:
			try:
				return int(m.group(1))
			except Exception:
				continue
	return None


def copy_file(src, dst):
	dst.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(src, dst)


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


def handle_multiple_csvs(videos, csvs, out_root, max_episodes=None):
	video_map = {v.stem: v for v in videos}
	created = 0
	for csv_path in sorted(csvs):
		name = csv_path.stem
		ep_id = extract_episode_from_name(name) or (created + 1)
		folder = out_root / f'episode_{ep_id}'
		folder.mkdir(parents=True, exist_ok=True)
		copy_file(csv_path, folder / csv_path.name)

		match_video = None
		if name in video_map:
			match_video = video_map[name]
		elif len(videos) == 1:
			match_video = videos[0]

		if match_video:
			copy_file(match_video, folder / match_video.name)

		print(f'Wrote episode {ep_id} to {folder}')
		created += 1
		if max_episodes and created >= max_episodes:
			break


def handle_single_csv(videos, csv_path, out_root, max_episodes=None):
	if pd is None:
		print('pandas not available; falling back to splitting csv by simple heuristics (rows)')
	df = None
	if pd is not None:
		df = pd.read_csv(csv_path)
	else:
		print('No pandas; copying CSV to episode_1 and copying video(s) there')
		folder = out_root / 'episode_1'
		folder.mkdir(parents=True, exist_ok=True)
		shutil.copy2(csv_path, folder / csv_path.name)
		if videos:
			shutil.copy2(videos[0], folder / videos[0].name)
		return

	ep_col = None
	for c in EPISODE_CANDIDATES:
		if c in df.columns:
			ep_col = c
			break

	if ep_col is None:
		frame_col = None
		for c in SINGLE_FRAME_CANDIDATES:
			if c in df.columns:
				frame_col = c
				break

		if frame_col is not None:
			print('No episode column found; splitting by contiguous frame gaps into episodes (heuristic)')
			frames = df[frame_col].astype(int).values
			gaps = [0] + [i for i in range(1, len(frames)) if frames[i] != frames[i-1] + 1]
			starts = gaps
			starts.append(len(frames))
			created = 0
			for i in range(len(starts)-1):
				s = starts[i]
				e = starts[i+1]-1
				ep_df = df.iloc[s:e+1]
				ep_id = created + 1
				folder = out_root / f'episode_{ep_id}'
				folder.mkdir(parents=True, exist_ok=True)
				ep_df.to_csv(folder / csv_path.name, index=False)
				if videos:
					shutil.copy2(videos[0], folder / videos[0].name)
				print('Wrote heuristic episode', ep_id)
				created += 1
				if max_episodes and created >= max_episodes:
					break
			return

		print('No episode or frame columns found in CSV; copying CSV and video into episode_1')
		folder = out_root / 'episode_1'
		folder.mkdir(parents=True, exist_ok=True)
		df.to_csv(folder / csv_path.name, index=False)
		if videos:
			shutil.copy2(videos[0], folder / videos[0].name)
		return

	groups = df.groupby(ep_col)
	created = 0
	for ep_val, ep_df in groups:
		created += 1
		folder = out_root / f'episode_{ep_val}'
		folder.mkdir(parents=True, exist_ok=True)
		ep_df.to_csv(folder / csv_path.name, index=False)

		start_frame = None
		end_frame = None
		for c in START_TIME_CANDIDATES:
			if c in ep_df.columns:
				start_frame = int(ep_df[c].min())
				break
		for c in END_TIME_CANDIDATES:
			if c in ep_df.columns:
				end_frame = int(ep_df[c].max())
				break

		for c in SINGLE_FRAME_CANDIDATES:
			if c in ep_df.columns:
				start_frame = int(ep_df[c].min())
				end_frame = int(ep_df[c].max())
				break

		if start_frame is not None and end_frame is not None and videos:
			video_path = videos[0]
			out_video = folder / video_path.name
			print(f'Extracting frames {start_frame}-{end_frame} to {out_video.name}')
			try:
				extract_video_segment(video_path, out_video, start_frame, end_frame)
			except Exception as e:
				print('Failed to extract segment:', e)
				shutil.copy2(video_path, out_video)
		else:
			if videos:
				shutil.copy2(videos[0], folder / videos[0].name)

		print(f'Wrote episode {ep_val} to {folder}')
		if max_episodes and created >= max_episodes:
			break


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--input_dir', required=True)
	p.add_argument('--output_dir', required=True)
	p.add_argument('--max_episodes', type=int, default=2)
	args = p.parse_args()

	input_dir = Path(args.input_dir)
	out_root = Path(args.output_dir)
	out_root.mkdir(parents=True, exist_ok=True)

	videos, csvs = find_files(input_dir)

	if not csvs:
		print('No CSV files found in', input_dir)
		sys.exit(1)

	if len(csvs) > 1:
		handle_multiple_csvs(videos, csvs, out_root, max_episodes=args.max_episodes)
	else:
		handle_single_csv(videos, csvs[0], out_root, max_episodes=args.max_episodes)


if __name__ == '__main__':
	main()

