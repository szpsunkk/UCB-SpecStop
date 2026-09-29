"""Run the target with the math-SDPA backend used in the Table 4 archive."""
from pathlib import Path
import runpy
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)
runpy.run_path(str(ROOT / "hardware/cloud_server.py"), run_name="__main__")
