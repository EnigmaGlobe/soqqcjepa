import pickle
import numpy as np
from pathlib import Path

def stats(arr):
    a = np.array(arr)
    if a.size == 0:
        return None
    if a.ndim >= 2:
        return a.mean(axis=(0,1)).tolist(), a.std(axis=(0,1)).tolist()
    else:
        return a.mean().tolist(), a.std().tolist()

def load_pickle(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

def print_stats(name, arr):
    if arr is None:
        print(f'{name}: missing')
        return
    print(f'{name} type: {type(arr)}')
    if isinstance(arr, dict):
        print(f'{name} keys: {list(arr.keys())[:10]}')
        # assume arr is {'train': {vid: arr, ...}, 'val': {...}} or similar
        for split in list(arr.keys()):
            try:
                sub = arr[split]
                # collect arrays inside sub
                arrays = []
                if isinstance(sub, dict):
                    for v in sub.values():
                        arrays.append(np.array(v))
                else:
                    arrays.append(np.array(sub))
                if arrays:
                    cat = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
                    m = cat.mean(axis=0).tolist()
                    s = cat.std(axis=0).tolist()
                    print(f'  split {split}: concatenated shape {cat.shape}, mean[0..3]={m[:4]} std[0..3]={s[:4]}')
                else:
                    print(f'  split {split}: no arrays')
            except Exception as e:
                print('  split', split, 'failed', e)
        return
    try:
        mean, std = stats(arr)
        print(f'{name} mean: {mean}')
        print(f'{name} std:  {std}')
    except Exception as e:
        print('Could not compute stats for', name, 'error', e)

def main():
    root = Path('checkpoints')
    # train action/proprio
    train_actions = root / 'local_action_meta.pkl'
    train_proprio = root / 'local_proprio_meta.pkl'
    train_slots = root / 'train_03_slots.pkl'

    print('=== TRAIN ===')
    if train_actions.exists():
        a = load_pickle(train_actions)
        print_stats('train_actions', a)
    else:
        print('train_actions missing')
    if train_proprio.exists():
        p = load_pickle(train_proprio)
        print_stats('train_proprio', p)
    else:
        print('train_proprio missing')

    # staged val files
    staged = Path('checkpoints/validation_staged')
    stages = [
        ('stage_0', 'train_03_recording_stage_0_exploration_actions.pkl', 'train_03_recording_stage_0_exploration_proprio.pkl'),
        ('stage_1', 'train_03_recording_stage_1_learning_actions.pkl', 'train_03_recording_stage_1_learning_proprio.pkl'),
        ('stage_2', 'train_03_recording_stage_2_converging_actions.pkl', 'train_03_recording_stage_2_converging_proprio.pkl'),
    ]

    for name, a_fn, p_fn in stages:
        print('\n===', name, '===')
        a_path = staged / a_fn
        p_path = staged / p_fn
        if a_path.exists():
            a = load_pickle(a_path)
            print_stats(f'{name}_actions', a)
        else:
            print(f'{name}_actions missing')
        if p_path.exists():
            pr = load_pickle(p_path)
            print_stats(f'{name}_proprio', pr)
        else:
            print(f'{name}_proprio missing')

if __name__ == '__main__':
    main()
