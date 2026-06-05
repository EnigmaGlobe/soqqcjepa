import torch
try:
    from torchvision.io import read_video
except Exception:
    read_video = None
import torchvision
import numpy as np
import pickle as pkl
from pathlib import Path
from torchvision import transforms as T
import sys

video_dir = sys.argv[1] if len(sys.argv) > 1 else 'testdata/train01'
out_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/train01_slots.pkl'

# build model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = torchvision.models.resnet18(pretrained=True)
modules = list(resnet.children())[:-1]  # remove fc
feat_extractor = torch.nn.Sequential(*modules).to(device).eval()
proj = torch.nn.Linear(512, 128).to(device).eval()

tf = T.Compose([
    T.ToPILImage(),
    T.Resize((196,196)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

p = Path(video_dir)
files = sorted([x for x in p.glob('**/*.mp4')])
if not files:
    print('No mp4 files found in', video_dir)
    raise SystemExit(1)

out = {'train':{}, 'val':{}}
for f in files:
    print('Processing', f)
    frames = []
    if read_video is not None:
        try:
            video_t, _, _ = read_video(str(f))
            if video_t is None or video_t.shape[0] == 0:
                frames = []
            else:
                # video_t: Tensor[T, H, W, C] uint8 RGB
                vid_np = video_t.numpy()
                frames = [vid_np[i] for i in range(vid_np.shape[0])]
        except Exception:
            frames = []
    if len(frames) == 0:
        # fallback to cv2 if available
        try:
            import cv2
            cap = cv2.VideoCapture(str(f))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame[:, :, ::-1])
            cap.release()
        except Exception:
            frames = []
    Tn = len(frames)
    if Tn == 0:
        slots = np.zeros((0,4,128), dtype=np.float32)
        out['train'][f.stem + '_pixels.mp4'] = slots
        continue
    batch_size = 32
    slots_list = []
    for i in range(0, Tn, batch_size):
        batch_frames = frames[i:i+batch_size]
        # for each frame, split into 4 patches
        patches = []
        for fr in batch_frames:
            h,w,_ = fr.shape
            hm = h//2
            wm = w//2
            quads = [fr[0:hm,0:wm], fr[0:hm,wm:w], fr[hm:h,0:wm], fr[hm:h,wm:w]]
            patches.extend(quads)
        # transform and stack
        tensors = [tf(p) for p in patches]
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = feat_extractor(x)  # (N,512,1,1)
            feats = feats.reshape(feats.shape[0], -1)
            feats = proj(feats)
        feats = feats.cpu().numpy()
        # reshape to (batch, 4, 128)
        nb = len(batch_frames)
        feats = feats.reshape(nb, 4, -1)
        slots_list.append(feats)
    slots = np.concatenate(slots_list, axis=0)
    out['train'][f.stem + '_pixels.mp4'] = slots.astype(np.float32)

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'wb') as f:
    pkl.dump(out, f)
print('Wrote', out_path)
