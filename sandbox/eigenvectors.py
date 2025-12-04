#%%
import os
import pickle
import sys
sys.path.append("/workspace/mta_vision_transformers/")
import time
from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import numpy as np
import einops
import torch
import torch.nn as nn
import torch.utils.data
from matplotlib import pyplot as plt
from ncut_pytorch import NCUT
from nystrom_ncut import KernelNCut, SampleConfig
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset
from torch.utils._pytree import tree_flatten, tree_unflatten
from transformers import ViTModel, ViTImageProcessor, ViTConfig

from core.monitor import Monitor
from dataset.construct import ImageDataset
from dataset.library import DATASETS
from infrastructure import utils
from infrastructure.settings import DEVICE, OUTPUT_DEVICE, DTYPE
from visualize.base import _visualize_cmap_with_values



base_model_name = "facebook/dino-vitb16"
image_size = 672
image_processor = ViTImageProcessor.from_pretrained(base_model_name)
image_processor.__dict__.update({
    "size": {"height": image_size, "width": image_size},
})

model = ViTModel.from_pretrained(base_model_name).to(DEVICE)
model.config.image_size = 672


class COCO2017Dataset(Dataset):
    def __init__(self, image_dir: str):
        with open("dataset/val2017.pkl", "rb") as fp:
            self.data: List[Tuple[str, List[str]]] = pickle.load(fp)
        self.image_dir: str = image_dir
        self.n_captions: torch.Tensor = torch.Tensor([len(l[1]) for l in self.data])
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        return image_processor(images=self.data[idx][0], return_tensors="pt")["pixel_values"][0]


if __name__ == "__main__":
    dataset_name, n_classes = DATASETS["Common"][1]
        
    # Ocean: 901085904
    # Rose: 100390212
    torch.set_printoptions(linewidth=400, sci_mode=False)
    
    bsz = 10
    dataloader = DataLoader(COCO2017Dataset("dataset/val2017"), batch_size=bsz, shuffle=False)
    images: torch.Tensor = next(iter(dataloader)).to(DEVICE)
    
    n_components = 10
    affinity_type = "rbf"
    num_sample = 2000
    
    transforms = OrderedDict([
        ("NCUT_fps", NCUT(
            num_eig=n_components,
            num_sample=num_sample,
            sample_method="farthest",
            distance=affinity_type,
        )),
        ("NCUT_random", NCUT(
            num_eig=n_components,
            num_sample=num_sample,
            sample_method="random",
            distance=affinity_type,
        )),
        ("KernelNCut_fps", KernelNCut(
            n_components=n_components,
            kernel_dim=4096,
            affinity_type=affinity_type,
            sample_config=SampleConfig(method="fps", num_sample=num_sample),
        )),
        ("KernelNCut_random", KernelNCut(
            n_components=n_components,
            kernel_dim=4096,
            affinity_type=affinity_type,
            sample_config=SampleConfig(method="random", num_sample=num_sample),
        ))
    ])
    
    with torch.no_grad():
        out: torch.Tensor = model.forward(images, interpolate_pos_encoding=True).last_hidden_state  # [bsz x (n + 1) x d]
        
        for name, transform in transforms.items():
            start_t = time.perf_counter()
            U = einops.rearrange(tree_flatten(transform.fit_transform(out.flatten(0, -2)))[0][0], "(bsz n) d -> bsz n d", bsz=bsz)
            end_t = time.perf_counter()
            print("=" * 120)
            print(f"{name}: {1000 * (end_t - start_t):.3f}ms")
            print("=" * 120)
            image_U = einops.rearrange(U[:, 1:, :], "bsz (h w) d -> bsz h w d", h=42, w=42)

            for eig_idx in range(n_components):
                _visualize_cmap_with_values(image_U[..., eig_idx], title=f"Eigenvector {eig_idx}", cmap="seismic")
            plt.show()
            plt.close()

            utils.empty_cache()

# %%
