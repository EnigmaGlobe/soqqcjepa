from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys, os

if len(sys.argv)<2:
    print('Usage: inspect_event_tags.py <event_file>')
    sys.exit(1)

ef = sys.argv[1]
if not os.path.exists(ef):
    print('Not found:', ef)
    sys.exit(2)

ea = EventAccumulator(ef)
ea.Reload()
print('Tags:', ea.Tags())
# print steps for scalars
scalars = ea.Tags().get('scalars', [])
for tag in scalars:
    vals = ea.Scalars(tag)
    if not vals:
        continue
    steps = [s.step for s in vals]
    print(f"Tag={tag}: steps {min(steps)}..{max(steps)} count={len(steps)}")
