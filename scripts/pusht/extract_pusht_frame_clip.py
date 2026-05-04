"""Extract an ordered Push-T frame clip from the Stable-WorldModel dataset cache.

This helper finds a Push-T rollout video inside the expected cache layout and
writes a short sequence of PNG frames that can be fed directly into the phase 2
pipeline script.

Expected dataset layout:
    ~/.stable_worldmodel/pusht_expert_train/videos/*.mp4
    ~/.stable_worldmodel/pusht_expert_val/videos/*.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def default_cache_dir() -> Path:
    """Return the default Stable-WorldModel cache root."""
    return Path.home() / ".stable_worldmodel"


def resolve_videos_dir(cache_dir: Path, dataset_name: str, split: str) -> Path:
    """Return the videos directory for a specific Push-T dataset split."""
    videos_dir = cache_dir / f"{dataset_name}_{split}" / "videos"
    if not videos_dir.exists():
        raise FileNotFoundError(
            "Push-T videos directory not found. Expected: "
            f"{videos_dir}\n"
            "Download and unpack the dataset under ~/.stable_worldmodel or pass --cache-dir."
        )
    return videos_dir


def choose_video(videos_dir: Path, video_name: str | None, video_index: int) -> Path:
    """Resolve one rollout video by explicit name or sorted index."""
    if video_name is not None:
        video_path = videos_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Requested video not found: {video_path}")
        return video_path

    candidates = sorted(videos_dir.glob("*_pixels.mp4"))
    if not candidates:
        raise FileNotFoundError(f"No Push-T videos found in: {videos_dir}")
    if video_index < 0 or video_index >= len(candidates):
        raise IndexError(f"Video index {video_index} is out of range for {len(candidates)} videos")
    return candidates[video_index]


def select_frame_indices(total_frames: int, start_frame: int, num_frames: int, stride: int) -> list[int]:
    """Build the ordered list of frame indices to extract."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")

    indices = [start_frame + offset * stride for offset in range(num_frames)]
    if not indices:
        raise ValueError("No frame indices were selected")
    if indices[-1] >= total_frames:
        raise ValueError(
            f"Requested frames {indices[0]}..{indices[-1]} exceed video length {total_frames}"
        )
    return indices


def probe_total_frames(video_path: Path) -> int:
    """Return the total number of frames in a rollout video using ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe did not return any video streams for {video_path}")
    total_frames = streams[0].get("nb_read_frames")
    if total_frames is None:
        raise RuntimeError(f"ffprobe could not determine frame count for {video_path}")
    return int(total_frames)


def extract_frames_with_ffmpeg(video_path: Path, frame_indices: list[int], output_dir: Path) -> None:
    """Extract the requested frame indices into numbered PNG files using ffmpeg."""
    output_pattern = output_dir / "%02d.png"
    select_expr = "+".join(f"eq(n\,{frame_index})" for frame_index in frame_indices)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select='{select_expr}'",
        "-vsync",
        "0",
        str(output_pattern),
        "-y",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for Push-T clip extraction."""
    parser = argparse.ArgumentParser(description="Extract a real Push-T frame clip for phase 2")
    parser.add_argument("--cache-dir", type=str, default=str(default_cache_dir()), help="Stable-WorldModel cache root")
    parser.add_argument("--dataset-name", type=str, default="pusht_expert", help="Dataset name prefix before _train/_val")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split to read from")
    parser.add_argument("--video-name", type=str, default=None, help="Specific rollout video filename, e.g. 0_pixels.mp4")
    parser.add_argument("--video-index", type=int, default=0, help="Sorted rollout index when --video-name is omitted")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to extract")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of consecutive frames to extract")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride between extracted images")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/pusht/sample_frames/pusht_real_clip",
        help="Directory where PNG frames and metadata will be written",
    )
    return parser.parse_args()


def main() -> int:
    """Extract frames from one Push-T rollout video into a phase-2-friendly folder."""
    args = parse_args()

    cache_dir = Path(args.cache_dir).expanduser()
    videos_dir = resolve_videos_dir(cache_dir, args.dataset_name, args.split)
    video_path = choose_video(videos_dir, args.video_name, args.video_index)

    total_frames = probe_total_frames(video_path)
    frame_indices = select_frame_indices(total_frames, args.start_frame, args.num_frames, args.stride)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extract_frames_with_ffmpeg(video_path, frame_indices, output_dir)

    metadata = {
        "video_path": str(video_path),
        "split": args.split,
        "dataset_name": args.dataset_name,
        "video_index": args.video_index,
        "selected_frame_indices": frame_indices,
        "total_frames": total_frames,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[extract] Video source: {video_path}")
    print(f"[extract] Total frames in source video: {total_frames}")
    print(f"[extract] Selected frame indices: {frame_indices}")
    print(f"[extract] Output directory: {output_dir}")
    print("[extract] You can now run phase 2 with --frames-dir pointing to this folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())