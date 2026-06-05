import pickle, os
from pathlib import Path

def print_slots_info(p):
    print("\n== Slots file:", p)
    if not os.path.exists(p):
        print("MISSING")
        return
    d = pickle.load(open(p, 'rb'))
    tr = d.get('train', {})
    print('train keys count:', len(tr))
    for k, v in list(tr.items())[:5]:
        try:
            print(k, getattr(v, 'shape', None))
        except Exception as e:
            print(k, 'error reading shape', e)
    # sample stats
    shapes = [getattr(v, 'shape', None) for v in tr.values()]
    print('unique shapes:', sorted(set(shapes)))


def print_meta_info(p):
    print("\n== Meta file:", p)
    if not os.path.exists(p):
        print('MISSING')
        return
    try:
        m = pickle.load(open(p, 'rb'))
        if isinstance(m, dict):
            print('dict keys sample:', list(m.keys())[:10])
            print('len:', len(m))
        else:
            try:
                print('type:', type(m), 'len:', len(m))
            except Exception:
                print('type:', type(m))
    except Exception as e:
        print('failed to load meta:', e)


if __name__ == '__main__':
    base = Path('checkpoints')
    slots_file = base / 'train01_slots_videosaur.pkl'
    slots_alt = base / 'train01_slots.pkl'
    slots_simple = base / 'train01_slots_simple.pkl'
    print_slots_info(str(slots_file))
    print_slots_info(str(slots_alt))
    print_slots_info(str(slots_simple))

    print_meta_info(str(base / 'local_action_meta.pkl'))
    print_meta_info(str(base / 'local_proprio_meta.pkl'))
    print_meta_info(str(base / 'local_state_meta.pkl'))

    print("\n== Checkpoints dir listing")
    if base.exists():
        for f in sorted(base.iterdir()):
            print(f.name, f.stat().st_size)
    else:
        print('checkpoints dir missing')

    print('\n== Lightning logs')
    logs = Path('lightning_logs')
    if logs.exists():
        print('lightning_logs exists, children:')
        for c in sorted(logs.iterdir()):
            print(' -', c.name)
    else:
        print('no lightning_logs dir')

    print('\nDone')
