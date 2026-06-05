import pandas as pd
p='testdata/1/data_training/observations_frame_train_01.csv'
try:
    df=pd.read_csv(p, nrows=5)
    print('COLUMNS:')
    print(list(df.columns))
    print('\nSAMPLE:')
    print(df.head().to_string(index=False))
except Exception as e:
    print('ERROR', e)
