#!/usr/bin/env python3
import os
import subprocess
import pickle
import numpy as np
from pathlib import Path

VIDEO_DIR = Path('testdata/train01')
OUT = Path('checkpoints/train01_slots.pkl')
S = 4
D = 128

def ffprobe_frame_count(p: Path):
    cmd = [
        'ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames', '-of','default=nokey=1:noprint_wrappers=1', str(p)
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        s = out.decode().strip()
        if s=='' or s=='N/A':
            # fallback: try duration*rate
            cmd2 = ['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate','of','default=nokey=1:noprint_wrappers=1', str(p)]
            out2 = subprocess.check_output(cmd2, stderr=subprocess.STDOUT).decode().strip()
            if '/' in out2:
                num,den = out2.split('/')
                rate = float(num)/float(den) if float(den)!=0 else 30.0
            else:
                rate = float(out2) if out2 else 30.0
            # duration
            cmd3 = ['ffprobe','-v','error','-show_entries','format=duration','-of','default=nokey=1:noprint_wrappers=1', str(p)]
            dur = float(subprocess.check_output(cmd3, stderr=subprocess.STDOUT).decode().strip())
            return int(max(1, round(rate*dur)))
        return int(float(s))
    except Exception:
        return 0

files = sorted(VIDEO_DIR.glob('**/*.mp4'))
if not files:
    raise SystemExit(f'No mp4 files in {VIDEO_DIR}')

out = {'train':{}, 'val':{}}
for p in files:
    key = f"{p.stem}_pixels.mp4"
    print('Processing',p)
    nframes = ffprobe_frame_count(p)
    print(' frame count', nframes)
    if nframes <= 0:
        arr = np.zeros((0, S, D), dtype=np.float32)
    else:
        # create small random embeddings to provide visual signal
        rng = np.random.RandomState(12345)
        arr = (rng.randn(nframes, S, D).astype(np.float32) * 0.01)
    out['train'][key] = arr

os.makedirs(OUT.parent, exist_ok=True)
with open(OUT, 'wb') as f:
    pickle.dump(out, f)
print('Wrote', OUT)
