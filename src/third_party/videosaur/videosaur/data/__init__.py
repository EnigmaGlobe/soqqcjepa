try:
	from src.third_party.videosaur.videosaur.data.datamodules import build
except Exception:  # noqa: BLE001
	def build(*args, **kwargs):  # type: ignore[no-redef]
		raise ImportError("videosaur.data.datamodules is optional for the phase 2 pipeline")

from src.third_party.videosaur.videosaur.data.utils import get_data_root_dir

__all__ = ["build", "get_data_root_dir"]
