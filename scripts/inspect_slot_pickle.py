#!/usr/bin/env python3
import pickle,os,sys
PK='checkpoints/train01_slots.pkl'
print('Opening',PK)
with open(PK,'rb') as f:
    data=pickle.load(f)
print('Top keys:', list(data.keys()))
train=data.get('train')
print('Type(train):',type(train))
# try to index a sample
sample=None
if isinstance(train, dict):
    keys=list(train.keys())
    print('train dict keys sample:', keys[:5])
    sample=train[keys[0]]
elif isinstance(train, list) or hasattr(train,'__len__'):
    try:
        sample=train[0]
    except Exception as e:
        print('Cannot index train list:',e)
        sample=None
else:
    print('train is unknown type')

print('Sample type:', type(sample))
if isinstance(sample, dict):
    print('Sample keys:', list(sample.keys()))
    for k in list(sample.keys()):
        try:
            v=sample[k]
            print(k, type(v))
            if hasattr(v,'shape'):
                print(' - shape', getattr(v,'shape'))
            elif isinstance(v,(list,tuple)):
                print(' - len', len(v))
        except Exception as e:
            print(' - failed to inspect',k,e)
else:
    print('Sample repr (trim):', repr(sample)[:500])
print('Done')
