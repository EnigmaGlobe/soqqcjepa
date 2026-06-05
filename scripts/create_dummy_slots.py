#!/usr/bin/env python3
"""Create dummy slot pickles from local MP4 files.

This script counts frames in each MP4 under a directory and saves
a pickle with structure matching the training code:
  {"train": {"<stem>_pixels.mp4": np.zeros((T, S, D))}, "val": {}}

Use when Videosaur or other encoders are unavailable locally.
"""
import argparse
from pathlib import Path
import pickle
import numpy as np
try:
    import cv2
except Exception:
    cv2 = None


def count_frames(path: Path) -> int:
    if cv2 is None:
        # OpenCV not available: try to infer frame count from nearby CSVs
        # Look for CSVs in the same directory and use the max step/frame index found
        import csv
        import glob
        csv_dir = path.parent
        pattern = str(csv_dir / "*.csv")
        csv_files = glob.glob(pattern)
        max_idx = -1
        step_keys = ("step", "step_idx", "step_index", "frame", "frame_idx")
        for cf in csv_files:
            try:
                with open(cf, newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    if not reader.fieldnames:
                        continue
                    # find matching step column
                    header_lower = [h.lower() for h in reader.fieldnames]
                    key = None
                    for k in step_keys:
                        if k in header_lower:
                            key = reader.fieldnames[header_lower.index(k)]
                            break
                    if key is None:
                        continue
                    for row in reader:
                        try:
                            v = row.get(key, "")
                            if v is None or v == "":
                                continue
                            idx = int(float(v))
                            if idx > max_idx:
                                max_idx = idx
                        except Exception:
                            continue
            except Exception:
                continue
        if max_idx >= 0:
            T = max_idx + 1
            print(f"Info: OpenCV missing — inferred T={T} from CSVs in {csv_dir}")
            return T
        print("Warning: OpenCV not installed and CSVs not usable; using T=1 for each video")
        return 1
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    cnt = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        cnt += 1
    cap.release()
    return cnt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_slots", type=int, default=4)
    p.add_argument("--slot_dim", type=int, default=128)
    args = p.parse_args()

    video_dir = Path(args.video_dir)
    files = sorted(video_dir.glob("**/*.mp4"))
    if not files:
        raise SystemExit(f"No mp4 files found in {video_dir}")

    out = {"train": {}, "val": {}}
    for f in files:
        T = count_frames(f)
        S = args.num_slots
        D = args.slot_dim
        print(f"Creating dummy slots for {f} -> T={T}, S={S}, D={D}")
        arr = np.zeros((T, S, D), dtype=np.float32)
        key = f"{f.stem}_pixels.mp4"
        out["train"][key] = arr

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(out, fh)

    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
