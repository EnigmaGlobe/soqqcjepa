import types


class EvalOnly:
    """Wrapper that mimics the tiny behaviour of the real EvalOnly helper.

    It simply forwards call to the underlying module in eval mode.
    """
    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs) if callable(self.module) else self.module

    def __getattr__(self, name):
        return getattr(self.module, name)
