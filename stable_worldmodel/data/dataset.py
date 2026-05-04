from torch.utils.data import Dataset

class VideoDataset(Dataset):
    """Stub: full implementation requires galilai-group/stable-worldmodel."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VideoDataset is not available in the bundled stable_worldmodel. "
            "Install the full package from https://github.com/galilai-group/stable-worldmodel for CLEVRER support."
        )
