import pickle, numpy as np, sys
p='checkpoints/train01_slots_proj_gpu.pkl'
if len(sys.argv)>1:
    p=sys.argv[1]
with open(p,'rb') as f:
    data=pickle.load(f)
print('top keys:', list(data.keys()))
train_keys = list(data.get('train',{}).keys())
print('num train keys', len(train_keys))
for k in train_keys[:10]:
    arr = data['train'][k]
    print(k, 'type', type(arr), 'shape', getattr(arr,'shape',None), 'dtype', getattr(arr,'dtype',None))
    if isinstance(arr, (list, tuple)):
        print('   list len', len(arr))
    else:
        if getattr(arr,'size',0)>0:
            print('   min', np.min(arr), 'max', np.max(arr), 'mean', np.mean(arr))
        else:
            print('   empty')
