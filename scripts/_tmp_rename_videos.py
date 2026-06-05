from pathlib import Path
p1=Path('testdata/1/episodes/episode_1/train_01_recording.mp4')
p2=Path('testdata/1/episodes/episode_2/train_01_recording.mp4')
for p, new in [(p1, 'episode_1.mp4'), (p2, 'episode_2.mp4')]:
    if p.exists():
        dst = p.parent / new
        p.rename(dst)
        print('Renamed', p, '->', dst)
    else:
        print('Not found', p)
