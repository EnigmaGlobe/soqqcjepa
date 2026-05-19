import sys
from pathlib import Path
import torch
import pickle

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.third_party.videosaur.videosaur import inference, configuration, models

VIDEO = Path("c:/new/soqqcjepa/testdata/train01/recording_compressed.mp4")
CKPT = Path("c:/new/soqqcjepa/checkpoints/pusht_videosaur_model.ckpt")
CFG = Path("c:/new/soqqcjepa/src/third_party/videosaur/configs/videosaur/pusht_dinov2_hf.yml")

print('VIDEO exists', VIDEO.exists())
print('CKPT exists', CKPT.exists())
print('CFG exists', CFG.exists())

try:
    model, conf = inference.load_model_from_checkpoint(str(CKPT), str(CFG))
    print('Model loaded, eval mode')
    model.eval()
except Exception as e:
    print('Failed to load model via inference:', e)
    raise

# Prepare inputs
try:
    inputs = inference.prepare_video(str(VIDEO), transfom_config=conf.model.get('input', None) if conf is not None else conf)
    print('Prepared inputs keys:', list(inputs.keys()))
    for k,v in inputs.items():
        try:
            print(k, getattr(v, 'shape', None))
        except Exception:
            pass
except Exception as e:
    print('Failed to prepare video inputs:', e)
    raise

with torch.no_grad():
    try:
        outputs = model(inputs)
        print('Model outputs keys:', list(outputs.keys()))
        # try aux_forward
        try:
            aux = model.aux_forward(inputs, outputs)
            print('Aux outputs keys:', list(aux.keys()))
        except Exception as e:
            print('aux_forward failed:', e)

        # if outputs contains 'state' or 'slots'
        if 'state' in outputs:
            s = outputs['state']
            print('state type', type(s), 'len', len(s))
            try:
                import numpy as np
                arr = s[0].cpu().numpy()
                print('state[0] shape', arr.shape, 'mean', arr.mean(), 'std', arr.std())
                pickle.dump({'slots': arr}, open('checkpoints/test_videosaur_slots.pkl','wb'))
                print('Wrote checkpoints/test_videosaur_slots.pkl')
            except Exception as e:
                print('Failed to dump state array:', e)
        elif 'slots' in outputs:
            s = outputs['slots']
            arr = s.cpu().numpy() if hasattr(s,'cpu') else s
            print('slots shape', getattr(arr,'shape',None))
            pickle.dump({'slots': arr}, open('checkpoints/test_videosaur_slots.pkl','wb'))
            print('Wrote checkpoints/test_videosaur_slots.pkl')
        else:
            print('No state/slots in outputs; keys:', outputs.keys())

    except Exception as e:
        print('Model forward failed:', e)
        raise

print('Done')
