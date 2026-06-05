import torch.nn as nn


class Embedder(nn.Module):
    """Simple linear embedder to mimic stable_worldmodel Embedder.

    This implementation lazily constructs a `nn.Linear` to project the
    last dimension of the input to `emb_dim`, so it is robust to callers
    that pass varying input shapes.
    """
    def __init__(self, in_chans=None, emb_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        # `self.net` will be created on first forward pass to match input dim
        self.net = None

    def forward(self, x):
        # x: (B, T, C_in) or (..., C_in)
        C_in = x.shape[-1]
        if self.net is None or getattr(self.net, 'in_features', None) != C_in:
            self.net = nn.Linear(C_in, self.emb_dim).to(x.device)
        return self.net(x)

