#!/usr/bin/env python3
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT='lightning_logs'
check_versions = sorted([d for d in os.listdir(ROOT) if d.startswith('version_')], key=lambda x: int(x.split('_')[1]))
want = ['train/loss','train/proprio_loss','train/pred_embedding_nan_count','train/pixels_embed_mean']

for v in check_versions:
    path = os.path.join(ROOT,v)
    # find event file
    ev_files = [f for f in os.listdir(path) if f.startswith('events.out.tfevents')]
    if not ev_files:
        continue
    ev_path = os.path.join(path, ev_files[0])
    try:
        ea = EventAccumulator(ev_path, size_guidance={
            'scalars': 0,
        })
        ea.Reload()
    except Exception as e:
        print(v, 'FAILED to load event file:', e)
        continue
    print('---', v, '---')
    for tag in want:
        try:
            if tag in ea.Tags().get('scalars', []):
                vals = ea.Scalars(tag)
                if vals:
                    last = vals[-1]
                    print(tag, 'step=', last.step, 'value=', last.value)
                else:
                    print(tag, 'no values')
            else:
                print(tag, 'missing')
        except Exception as e:
            print(tag, 'err', e)
print('done')
