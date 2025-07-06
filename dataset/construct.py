from typing import Callable, Optional, Tuple

import datasets
import torch
from PIL.ImageFile import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from infrastructure.settings import DEVICE
from modeling.base_vit import OpenCLIPViT


class ImageDataset(Dataset):
    default_transform: Callable[[ImageFile], torch.Tensor] = transforms.Compose((
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ))
    
    def __init__(self,
                 dataset_name: str,
                 split: str = "train",
                 return_original_image: bool = False
    ):
        self.data = datasets.load_dataset(dataset_name, split=split, trust_remote_code=True)
        if return_original_image:
            self.transform_list = [ImageDataset.default_transform, OpenCLIPViT.preprocess_func]
        else:
            self.transform_list = [OpenCLIPViT.preprocess_func]

    def __len__(self) -> int:
        return self.data.num_rows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return tuple(
            transform(self.data[idx]["image"]).to(DEVICE)
            for transform in self.transform_list
        )




