#!/usr/bin/env python3
import os, csv, math
ROOT='lightning_logs'

def is_float(x):
    try:
        v=float(x)
        return True
    except:
        return False

for version in sorted(os.listdir(ROOT)):
    version_path=os.path.join(ROOT,version)
    csv_path=os.path.join(version_path,'metrics.csv')
    if not os.path.isfile(csv_path):
        continue
    print(f"\n--- {version} ({csv_path}) ---")
    cols=None
    nan_counts=None
    total=0
    with open(csv_path,'r',newline='') as f:
        reader=csv.DictReader(f)
        cols=reader.fieldnames
        nan_counts={c:0 for c in cols}
        finite_counts={c:0 for c in cols}
        for i,row in enumerate(reader):
            total+=1
            for c in cols:
                v=row.get(c,'')
                if v is None or v=='' or not is_float(v):
                    nan_counts[c]+=1
                else:
                    try:
                        fv=float(v)
                        if math.isfinite(fv):
                            finite_counts[c]+=1
                        else:
                            nan_counts[c]+=1
                    except:
                        nan_counts[c]+=1
            if i>=1000:
                break
    print(f"Rows scanned: {total}")
    # report columns where all values are non-finite
    all_nan=[c for c in cols if nan_counts.get(c,0)>=max(1,total)]
    some_nan=[c for c in cols if 0<nan_counts.get(c,0)<(total if total>0 else 1)]
    print(f"Columns all non-finite or missing: {all_nan}")
    print(f"Columns with some non-finite values: {some_nan}")
    # show sample header and first non-empty row
    with open(csv_path,'r',newline='') as f:
        first_line=f.readline()
        second_line=f.readline()
    print('Header:', first_line.strip())
    print('Sample row:', second_line.strip())
print('\nScan complete.')
