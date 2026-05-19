import pickle, os
pfn = 'checkpoints/train01_slots_real4.pkl'
print('exists', os.path.exists(pfn))
with open(pfn,'rb') as f:
    obj = pickle.load(f)
print('top keys:', list(obj.keys()))
for split in obj:
    print('split', split, 'len', len(obj[split]))
    for k,v in list(obj[split].items())[:5]:
        print(' ', k, getattr(v,'shape', None))
