import pickle
import numpy as np

path = 'checkpoints/local_proprio_meta.pkl'
obj = pickle.load(open(path,'rb'))
print('top keys', list(obj.keys()))
d = obj['train']
for k,v in list(d.items())[:5]:
    a = np.asarray(v)
    print('video',k,'shape',a.shape,'anynan',np.isnan(a).any(),'anyinf',np.isinf(a).any())
    std = np.nanstd(a,axis=0)
    print(' std per-dim:', std)
    print(' zeros in std:', (std==0))
    # print sample rows with NaN
    nan_rows = np.where(np.isnan(a).any(axis=1))[0]
    print(' rows with NaN count:', len(nan_rows))
    if len(nan_rows)>0:
        print(' first NaN row sample:', a[nan_rows[:3]])
    break
