import os
from pathlib import Path
p = Path('testdata/train01/recording_compressed.mp4')
print('Path:', p)
print('Exists:', p.exists())
if p.exists():
    print('Size:', p.stat().st_size)

# torchvision
try:
    from torchvision.io import read_video
    v, a, info = read_video(str(p))
    print('torchvision frames:', None if v is None else v.shape)
except Exception as e:
    print('torchvision error:', e)

# cv2
try:
    import cv2
    cap = cv2.VideoCapture(str(p))
    cnt = 0
    ok = True
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cnt += 1
    cap.release()
    print('cv2 frames count:', cnt)
except Exception as e:
    print('cv2 error:', e)

# imageio
try:
    import imageio
    r = imageio.get_reader(str(p))
    cnt = 0
    for _ in r:
        cnt += 1
        if cnt>10000:
            break
    r.close()
    print('imageio frames count:', cnt)
except Exception as e:
    print('imageio error:', e)

print('Done')
