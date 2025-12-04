import numpy as np
import os
import sys
import torch


# SEED: int = 100390212
SEED: int = 1149496617
# SEED: int = None
# SEED: int = int(np.random.randint(0, np.iinfo(np.int32).max))   # 27579224
print(f"Seed: {SEED}")

DEVICE: str = "cuda:0"
OUTPUT_DEVICE: str = "cpu"
DTYPE: torch.dtype = torch.float32
PROJECT_NAME: str = "mta_vision_transformers"
PROJECT_PATH: str = os.getcwd()[:os.getcwd().find(PROJECT_NAME)] + PROJECT_NAME

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)
torch.set_printoptions(precision=6, sci_mode=False, linewidth=400)
os.chdir(PROJECT_PATH)
sys.path.append(PROJECT_PATH)




