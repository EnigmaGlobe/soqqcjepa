import pickle, numpy as np, sys
p='checkpoints/train01_slots_resnet.pkl'
with open(p,'rb') as f:
    data=pickle.load(f)
train=data.get('train',{})
all_arrays=[]
for k,v in train.items():
    arr=np.asarray(v,dtype=np.float32)
    if arr.size>0:
        # reshape to (T*S, D)
        all_arrays.append(arr.reshape(-1,arr.shape[-1]))
if not all_arrays:
    print('No data')
    sys.exit(0)
A=np.concatenate(all_arrays,axis=0)
print('total_frames_slots:',A.shape[0])
print('dim:',A.shape[1])
print('abs_max:',float(np.max(np.abs(A))))
print('mean:',float(np.mean(A)))
print('nonzero count:',int((A!=0).sum()))
print('nonzero ratio:',float((A!=0).sum())/A.size)
