from typing import Callable, Optional, Tuple

import datasets
import torch
from PIL.ImageFile import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from infrastructure.settings import DEVICE


class ImageDataset(Dataset):
    default_transform: Callable[[ImageFile], torch.Tensor] = transforms.Compose((
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ))
    
    def __init__(self,
                 dataset_name: str,
                 transform: Callable[[ImageFile], torch.Tensor] = None,
                 split: str = "train",
                 return_original_image: bool = False
    ):
        self.data = datasets.load_dataset(dataset_name)[split]
        if transform is None:
            self.transform = ImageDataset.default_transform
        else:
            self.transform = transform
        self.return_original_image = return_original_image

    def __len__(self) -> int:
        return self.data.num_rows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.return_original_image:
            transform_list = [ImageDataset.default_transform, self.transform]
        else:
            transform_list = [self.transform]
        return tuple(transform(self.data[idx]["image"]).to(DEVICE) for transform in transform_list)




