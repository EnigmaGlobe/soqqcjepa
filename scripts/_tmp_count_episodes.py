import pandas as pd
p='testdata/1/data_training/observations_frame_train_01.csv'
df=pd.read_csv(p, usecols=['episode_id'])
print('unique episodes:', sorted(df['episode_id'].unique())[:20])
print('count per episode:')
print(df['episode_id'].value_counts().sort_index())
