import cv2
import os
from pathlib import Path
p = Path('C:/new/soqqcjepa/testdata/train01/recording_compressed.mp4')
print('exists:', p.exists())
print('size bytes:', p.stat().st_size if p.exists() else None)

# OpenCV
try:
    cap = cv2.VideoCapture(str(p))
    cnt = 0
    ok = True
    while ok:
        ok, frame = cap.read()
        if ok:
            cnt += 1
    cap.release()
    print('cv2 frames:', cnt)
except Exception as e:
    print('cv2 error:', e)

# torchvision
try:
    from torchvision.io import read_video
    video, audio, info = read_video(str(p))
    print('torchvision read_video frames:', len(video))
except Exception as e:
    print('torchvision error:', e)

# ffmpeg probe via imageio (if available)
try:
    import imageio
    reader = imageio.get_reader(str(p))
    n = 0
    for _ in reader:
        n += 1
    reader.close()
    print('imageio frames:', n)
except Exception as e:
    print('imageio error:', e)
