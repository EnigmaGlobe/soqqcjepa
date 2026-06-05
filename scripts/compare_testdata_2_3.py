import csv
import math
from pathlib import Path

def col_stats(path):
    path = Path(path)
    with path.open('r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [[] for _ in header]
        for row in reader:
            if len(row) != len(header):
                continue
            for i, v in enumerate(row):
                try:
                    cols[i].append(float(v))
                except:
                    pass
    stats = {}
    for h, c in zip(header, cols):
        arr = c
        if not arr:
            stats[h] = (None, None)
        else:
            n = len(arr)
            mean = sum(arr)/n
            var = sum((x-mean)**2 for x in arr)/n
            stats[h] = (mean, math.sqrt(var))
    return stats

def compare():
    base2 = Path('testdata/validation/2/data_training')
    base3 = Path('testdata/validation/3/data_training')
    files = [
        (base2 / 'actions_frame_train_02.csv', base3 / 'actions_frame_train_03.csv'),
        (base2 / 'observations_frame_train_02.csv', base3 / 'observations_frame_train_03.csv')
    ]
    keys_of_interest = ['action_x','action_y','agent_pos_x','agent_pos_y','agent_pos_z','block_pos_x','block_pos_y','block_pos_z']
    for f2, f3 in files:
        print('\nComparing:', f2.name, 'vs', f3.name)
        s2 = col_stats(f2)
        s3 = col_stats(f3)
        for k in keys_of_interest:
            if k in s2 and k in s3:
                m2,sd2 = s2[k]
                m3,sd3 = s3[k]
                print(f'{k}: 2 mean={m2!s} std={sd2!s}  | 3 mean={m3!s} std={sd3!s}')

if __name__ == '__main__':
    compare()
