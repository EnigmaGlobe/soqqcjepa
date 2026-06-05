#!/usr/bin/env python3
try:
    import imageio
except Exception:
    imageio = None
try:
    import cv2
except Exception:
    cv2 = None

p='testdata/train01/recording_compressed.mp4'
print('Path:',p)
try:
    r=imageio.get_reader(p)
    cnt=0
    for _ in r:
        cnt+=1
        if cnt>5:
            break
    print('imageio sample frames:',cnt)
except Exception as e:
    print('imageio failed:',e)

if cv2 is None:
    print('cv2 not installed')
else:
    try:
        cap=cv2.VideoCapture(p)
        cnt=0
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            cnt+=1
            if cnt>5:
                break
        cap.release()
        print('cv2 sample frames:',cnt)
    except Exception as e:
        print('cv2 failed:',e)
