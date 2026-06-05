import argparse
import os
try:
    # Prefer tensorboard's summary_iterator
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    def summary_iter_from_file(path):
        ea = EventAccumulator(path)
        ea.Reload()
        # build list of (step, tag->value)
        items = []
        for tag in ea.Tags().get('scalars', []):
            scalars = ea.Scalars(tag)
            for s in scalars:
                items.append((s.step, tag, s.value))
        return items
    HAS_EVENT_ACC = True
except Exception:
    try:
        from torch.utils.tensorboard.summary_iterator import summary_iterator
        HAS_EVENT_ACC = False
    except Exception:
        raise
import csv


def find_event_file(logdir):
    # find first events.out.* file recursively
    for root, dirs, files in os.walk(logdir):
        for f in files:
            if f.startswith('events.out.'):
                return os.path.join(root, f)
    return None


def export(logdir, out_path):
    ef = find_event_file(logdir)
    if ef is None:
        raise SystemExit(f'No events file found under {logdir}')
    print(f'Using events file: {ef}')

    # collect tags and values per step
    data = {}  # step -> {tag: value}
    tags = set()
    if 'HAS_EVENT_ACC' in globals() and HAS_EVENT_ACC:
        items = summary_iter_from_file(ef)
        for step, tag, val in items:
            tags.add(tag)
            data.setdefault(step, {})[tag] = val
    else:
        for e in summary_iterator(ef):
            step = getattr(e, 'step', None)
            if not hasattr(e, 'summary') or e.summary is None:
                continue
            for v in e.summary.value:
                tag = v.tag
                val = None
                if v.HasField('simple_value'):
                    val = v.simple_value
                else:
                    continue
                tags.add(tag)
                data.setdefault(step, {})[tag] = val

    tags = sorted(tags)
    steps = sorted(data.keys())

    # write wide CSV: step, tag1, tag2...
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        header = ['step'] + tags
        w.writerow(header)
        for s in steps:
            row = [s]
            rowd = data.get(s, {})
            for t in tags:
                row.append(rowd.get(t, ''))
            w.writerow(row)
    print(f'Wrote CSV to {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', default='lightning_logs_tb')
    p.add_argument('--out', default='checkpoints/metrics.csv')
    args = p.parse_args()
    export(args.logdir, args.out)
