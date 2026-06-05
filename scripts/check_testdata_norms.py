import csv
import math
from pathlib import Path

def col_stats(path):
    path = Path(path)
    if not path.exists():
        print(f'Missing {path}')
        return
    with path.open('r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [[] for _ in header]
        for row in reader:
            if len(row) != len(header):
                # skip malformed
                continue
            for i, v in enumerate(row):
                try:
                    cols[i].append(float(v))
                except:
                    cols[i].append(float('nan'))

    stats = []
    for c in cols:
        arr = [x for x in c if not math.isnan(x)]
        if not arr:
            stats.append((None, None))
        else:
            n = len(arr)
            mean = sum(arr)/n
            var = sum((x-mean)**2 for x in arr)/n
            stats.append((mean, math.sqrt(var)))
    print(f'File: {path}')
    for h, (m, s) in zip(header, stats):
        print(f'  {h}: mean={m}, std={s}')

def main():
    base = Path('testdata/validation/2/data_training')
    act = base / 'actions_frame_train_02.csv'
    obs = base / 'observations_frame_train_02.csv'
    col_stats(act)
    print('')
    col_stats(obs)

if __name__ == '__main__':
    main()
