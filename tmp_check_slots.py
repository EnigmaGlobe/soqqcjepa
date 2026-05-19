import pickle, numpy as np
p=pickle.load(open('checkpoints/train01_slots_final.pkl','rb'))
print('top', list(p.keys()))
print('train', list(p['train'].keys()))
for k,a in p['train'].items():
    arr = np.asarray(a)
    print(k, arr.shape)
