"""Minimal shim of `stable_pretraining` to allow local training runs.

This provides a tiny `Module`, `Manager`, and `backbone.EvalOnly` used by
the project's training script. It's a light-weight compatibility layer
for running on local datasets without the full external dependency.
"""
from types import SimpleNamespace
from .backbone import EvalOnly


class Module:
    def __init__(self, model, forward, optim):
        self.model = model
        self.forward = forward
        self.optim = optim or {}


class Manager:
    def __init__(self, trainer, module, data, ckpt_path, seed=42):
        self.trainer = trainer
        self.module = module
        self.data = data
        self.ckpt_path = ckpt_path
        self.seed = seed

    def __call__(self):
        # Lightweight manager: wrap module.model and module.forward into a
        # PyTorch LightningModule and run trainer.fit.
        import lightning as pl
        import torch

        class _Wrapped(pl.LightningModule):
            def __init__(self, module: Module):
                super().__init__()
                self.module = module
                # Register the underlying model as a submodule so its parameters
                # are discovered by Lightning and optimizers can be created.
                try:
                    self.model = module.model
                except Exception:
                    self.model = None

            def training_step(self, batch, batch_idx):
                # Create a lightweight proxy that provides the minimal API the
                # project's `forward` function expects (model, training flag, and
                # a `log_dict` method). Use the LightningModule's `log_dict` to
                # perform actual logging.
                proxy = SimpleNamespace()
                proxy.model = self.module.model
                proxy.training = True
                def _log_dict(d, on_step=True, sync_dist=True):
                    try:
                        self.log_dict(d, on_step=on_step, sync_dist=sync_dist)
                    except Exception:
                        pass

                proxy.log_dict = _log_dict
                out = self.module.forward(proxy, batch, stage='train')
                # `out` may be a dict-like object containing 'loss'
                loss = None
                if isinstance(out, dict) and 'loss' in out:
                    loss = out['loss']
                elif hasattr(out, 'get') and out.get('loss') is not None:
                    loss = out.get('loss')
                if loss is None:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                self.log('train/loss', loss)
                return loss

            def validation_step(self, batch, batch_idx):
                proxy = SimpleNamespace()
                proxy.model = self.module.model
                proxy.training = False
                def _log_dict(d, on_step=True, sync_dist=True):
                    try:
                        self.log_dict(d, on_step=on_step, sync_dist=sync_dist)
                    except Exception:
                        pass
                proxy.log_dict = _log_dict
                out = self.module.forward(proxy, batch, stage='val')
                loss = None
                if isinstance(out, dict) and 'loss' in out:
                    loss = out['loss']
                if loss is None:
                    return None
                self.log('val/loss', loss)
                return loss

            def configure_optimizers(self):
                import torch.optim as optim
                # Create simple optimizers; fallback to Adam on all parameters
                if not self.module.optim:
                    return optim.Adam(self.parameters(), lr=1e-3)
                # If optim mapping present, use the first lr
                lrs = [v.get('optimizer', {}).get('lr', 1e-3) for v in self.module.optim.values()]
                return optim.Adam(self.parameters(), lr=lrs[0] if lrs else 1e-3)

        wrapped = _Wrapped(self.module)
        self.trainer.fit(wrapped, getattr(self.data, 'train', self.data), getattr(self.data, 'val', None))
        try:
            self.trainer.save_checkpoint(self.ckpt_path)
        except Exception:
            pass


# Minimal data submodule
class data:
    class DataModule(SimpleNamespace):
        def __init__(self, train=None, val=None):
            super().__init__(train=train, val=val)


__all__ = ["Module", "Manager", "backbone", "data"]
