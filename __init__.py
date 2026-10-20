"""ComfyUI entry point: this repository is the MotionBridge custom node pack."""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.append(_root)  # appended, so the repo's folders never shadow another package

from motionbridge.comfy.extension import comfy_entrypoint  # noqa: E402

WEB_DIRECTORY = './web'
__all__ = ['comfy_entrypoint', 'WEB_DIRECTORY']
