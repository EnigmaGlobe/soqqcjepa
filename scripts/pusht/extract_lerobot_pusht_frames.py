"""Download one episode from lerobot/pusht (HuggingFace) and extract frames.

The lerobot/pusht dataset is public — no authentication required.
Video layout on HF: videos/chunk-000/observation.image/episode_{N:06d}.mp4

Usage:
    python scripts/pusht/extract_lerobot_pusht_frames.py
    python scripts/pusht/extract_lerobot_pusht_frames.py --episode 5 --start-frame 10 --num-frames 5
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "lerobot/pusht"
REPO_TYPE = "dataset"
# All 206 episodes are concatenated into one 6.5 MB chunk file (XetHub storage)
CHUNK_PATH = "videos/observation.image/chunk-000/file-000.mp4"


def download_chunk_video(cache_dir: Path) -> Path:
    """Download the single chunk MP4 containing all episodes and return its local path."""
    cached = cache_dir / "videos" / "observation.image" / "chunk-000" / "file-000.mp4"
    if cached.exists():
        print(f"Using cached video: {cached}")
        return cached
    print(f"Downloading {CHUNK_PATH} from {REPO_ID} (~6.5 MB) ...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=CHUNK_PATH,
        local_dir=str(cache_dir),
    )
    return Path(local_path)


def probe_total_frames(video_path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def extract_frames(video_path: Path, frame_indices: list[int], out_dir: Path) -> list[Path]:
    """Extract specific frames from a video using ffmpeg select filter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, idx in enumerate(frame_indices):
        out_path = out_dir / f"frame_{i:04d}.png"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"select=eq(n\\,{idx})",
                "-vsync", "vfr",
                "-frames:v", "1",
                str(out_path),
            ],
            capture_output=True, check=True,
        )
        paths.append(out_path)
        print(f"  Extracted frame {idx} -> {out_path.name}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from lerobot/pusht on HuggingFace")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to extract (0-25649)")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to extract")
    parser.add_argument("--stride", type=int, default=3, help="Stride between frames")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scripts/pusht/sample_frames/pusht_lerobot_clip"),
        help="Output directory for PNG frames",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "lerobot_pusht",
        help="Local cache directory for downloaded videos",
    )
    args = parser.parse_args()

    video_path = download_chunk_video(args.cache_dir)
    print(f"Video at: {video_path}")

    total = probe_total_frames(video_path)
    print(f"Total frames: {total}")

    indices = [args.start_frame + i * args.stride for i in range(args.num_frames)]
    if indices[-1] >= total:
        raise ValueError(
            f"Requested frames {indices[0]}..{indices[-1]} exceed video length {total}. "
            f"Try --start-frame 0 --stride 1."
        )

    paths = extract_frames(video_path, indices, args.out_dir)
    print(f"\nDone. {len(paths)} frames written to: {args.out_dir}")
    print("Run phase 2 with:")
    print(f"  python scripts/pusht/phase2_client_integration_smoke.py --frames-dir {args.out_dir}")


if __name__ == "__main__":
    main()
