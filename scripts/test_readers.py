import imageio, os
p='testdata/train01/recording_compressed.mp4'
print('exists', os.path.exists(p))
try:
    r=imageio.get_reader(p)
    cnt=0
    for f in r:
        cnt+=1
        if cnt%10000==0:
            print('read',cnt)
        if cnt>100:
            break
    r.close()
    print('imageio read count preview',cnt)
except Exception as e:
    print('imageio failed', e)

try:
    from torchvision.io import read_video
    v,_,_ = read_video(p)
    print('torchvision read_video frames', len(v))
except Exception as e:
    print('torchvision failed', e)

try:
    import cv2
    cap=cv2.VideoCapture(p)
    cnt=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        cnt+=1
        if cnt>100: break
    cap.release()
    print('cv2 read count preview',cnt)
except Exception as e:
    print('cv2 failed', e)
