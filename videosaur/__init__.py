from pathlib import Path
import sys

_THIRD_PARTY_ROOT = Path(__file__).resolve().parents[1] / "src" / "third_party"
if str(_THIRD_PARTY_ROOT) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY_ROOT))

__path__ = [str(_THIRD_PARTY_ROOT / "videosaur")]
