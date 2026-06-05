import sys, traceback, torch
from src.world_models.dinowm_causal_AP_node import CausalWM_AP
p='checkpoints/local_real_train_epoch_5_object.ckpt'
print('path', p)
print('exists', __import__('os').path.exists(p))
try:
    sys.setrecursionlimit(20000)
    print('adding safe global', CausalWM_AP)
    with torch.serialization.safe_globals([CausalWM_AP]):
        obj = torch.load(p, map_location='cpu', weights_only=False)
    print('loaded type', type(obj))
    try:
        sd = obj.state_dict()
        print('got state_dict keys sample:', list(sd.keys())[:20])
        torch.save(sd, 'checkpoints/local_real_train_epoch_5_state_dict.ckpt')
        print('saved state_dict')
    except Exception as e:
        print('object has no state_dict', e)
except Exception as e:
    traceback.print_exc()
    print('failed to load checkpoint')
