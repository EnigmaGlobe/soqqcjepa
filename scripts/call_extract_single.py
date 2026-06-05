from pathlib import Path
from scripts.extract_slots_from_videos_local import extract_slots_for_file
p=Path('testdata/train01/recording_compressed.mp4')
print('path exists', p.exists())
res = extract_slots_for_file(None, p, device='cpu')
print('result type', type(res), getattr(res,'shape',None))
if hasattr(res,'size') and res.size>0:
    import numpy as np
    print('min,max,mean', np.min(res), np.max(res), np.mean(res))
else:
    print('empty result')
