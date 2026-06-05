import pickle

p = 'checkpoints/train_03_slots.pkl'
with open(p, 'rb') as f:
    d = pickle.load(f)

print('keys:', list(d.keys()))
for k, v in d.items():
    try:
        print(k, 'shape=', getattr(v, 'shape', None), 'type=', type(v))
    except Exception as e:
        print('key', k, 'inspect error', e)

if 'train' in d:
    print('\n--- train keys ---')
    for k2, v2 in d['train'].items():
        print(k2, '->', getattr(v2, 'shape', None), type(v2))

if 'val' in d:
    print('\n--- val keys ---')
    for k2, v2 in d['val'].items():
        print(k2, '->', getattr(v2, 'shape', None), type(v2))
