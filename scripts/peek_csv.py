import csv,sys
p = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv)>2 else 50
with open(p, newline='') as fh:
    r = csv.reader(fh)
    for i,row in enumerate(r):
        print(','.join(row))
        if i+1>=N:
            break
