import argparse
import sys
import importlib

# Wrap ArgumentParser.add_argument to avoid help-string type errors
_old_add = argparse.ArgumentParser.add_argument

def _safe_add(self, *args, **kwargs):
    try:
        return _old_add(self, *args, **kwargs)
    except Exception:
        if 'help' in kwargs and not isinstance(kwargs['help'], str):
            kwargs['help'] = str(kwargs['help'])
            try:
                return _old_add(self, *args, **kwargs)
            except Exception:
                return None
        return None

argparse.ArgumentParser.add_argument = _safe_add

sys.path.insert(0, r"C:\soqqle\soqqcjepa")
mod = importlib.import_module('src.train.train_causalwm_AP_node_pusht_slot')
mod.run()
