import pickle
from abc import abstractmethod
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple, Union
from PIL import Image

import datasets
import numpy as np
import torch
import torch.nn.functional as Fn
from tensordict import TensorDict
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler, Sampler, default_collate
from torch.utils._pytree import tree_flatten, tree_unflatten

from infrastructure import utils
from infrastructure.settings import DEVICE
from dataset.library import DATASETS
from modeling.base_vit import OpenCLIPViT


# COCO-2017 dataset

class BaseDataset(Dataset):
    mode_transforms: Dict[str, Callable[[Any], torch.Tensor]] = {
        "image": OpenCLIPViT.preprocess_func,
        "text": OpenCLIPViT.tokenizer_func,
    }
    
    def __init__(self):
        super().__init__()
        self._cache: TensorDict = TensorDict()
        self.n_captions: torch.Tensor = None
        
    def load_cache(self, d: Dict[str, Union[str, torch.Tensor]]) -> None:
        flattened_v, argspec = tree_flatten(d)
        flattened_v: List[torch.Tensor] = [
            torch.load(v, map_location=DEVICE) if isinstance(v, str) else v
            for v in flattened_v
        ]
        self._cache = TensorDict(tree_unflatten(flattened_v, argspec)).auto_batch_size_(batch_dims=1)
    
    @abstractmethod
    def get_idx(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        return self.get_idx(idx), ({} if self._cache.is_empty() else self._cache[idx].to_dict())


class COCO2017Dataset(BaseDataset):
    def __init__(self, image_dir: str):
        BaseDataset.__init__(self)
        with open("dataset/val2017.pkl", "rb") as fp:
            self.data: List[Tuple[str, List[str]]] = pickle.load(fp)
        self.image_dir: str = image_dir
        self.n_captions: torch.Tensor = torch.Tensor([len(l[1]) for l in self.data])
        
    def __len__(self) -> int:
        return len(self.data)

    def get_idx(self, idx: int) -> Dict[str, torch.Tensor]:
        image, captions = self.data[idx]
        return {
            "image": OpenCLIPViT.preprocess_func(image).to(DEVICE),
            "text": OpenCLIPViT.tokenizer_func(captions).to(DEVICE),
        }


# COCODataset class with text captions
class ImageTextDataset(BaseDataset):
    def __init__(self,
                 dataset_name: str,
                 keys: Dict[str, str],
                 split: str = "train",
    ):
        BaseDataset.__init__(self)
        self.data = datasets.load_dataset(dataset_name, split=split, trust_remote_code=True)
        self.keys = keys

    def __len__(self) -> int:
        return self.data.num_rows

    def get_idx(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            mode: transform(self.data[idx][self.keys[mode]]).to(DEVICE)
            for mode, transform in ImageTextDataset.mode_transforms.items()
        }


DEFAULT_DATASET = COCO2017Dataset("dataset/val2017")
# DEFAULT_DATASET = ImageTextDataset(DATASETS["Common"][1][0], keys={"image": "image", "text": "caption"}, split="train")
def run_retrieval_evaluation(
    model: OpenCLIPViT,
    dataset: BaseDataset = DEFAULT_DATASET,
    k_values: List[int] = [1, 5, 10],
    batch_size: int = 32,
    return_similarity_matrix: bool = False,
) -> Tuple[torch.Tensor, TensorDict]:
    """
    Compute retrieval metrics for the validation dataset.
    
    Args:
    - model: OpenCLIP model
    - dataset: Validation DataLoader
    - device: Computation device
    
    Returns:
    - dict: Retrieval metrics
    """
    utils.reset_seed()
    
    # Collect image and text embeddings
    model.eval()
    
    def get_retrieval(S: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:             # [N x M]
        max_k = max(k_values)
        correct = mask[torch.arange(S.shape[0])[:, None], torch.topk(S, k=max_k, dim=1).indices]
        return {
            f"R@{k}": torch.mean(torch.any(correct[:, :k], dim=1).to(torch.float)).item()
            for k in k_values
        }
    
    n_captions: List[int] = []
    def collate_fn(data: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        data_list, cache_list = zip(*data)
        collated_data: Dict[str, torch.Tensor] = {
            "image": torch.stack([d["image"] for d in data_list], dim=0),
            "text": torch.cat([d["text"] for d in data_list], dim=0)
        }
        n_captions.extend((d["text"].shape[0] for d in data_list))
        collated_cache: Dict[str, torch.Tensor] = default_collate(cache_list)
        return collated_data, collated_cache
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    embedding_lists: Dict[str, List[torch.Tensor]] = {"image": [], "text": []}
    
    with torch.no_grad():
        for data, cached_data in tqdm(dataloader):
            model.load_cache(cached_data)
        
            data: Dict[str, torch.Tensor]
            for k, t in data.items():
                embedding_lists.setdefault(k, []).append(getattr(model.model, f"encode_{k}")(t, normalize=True))

            utils.empty_cache()
            
    # Concatenate embeddings
    normalized_embeddings: Dict[str, torch.Tensor] = {k: torch.cat(v, dim=0) for k, v in embedding_lists.items()}
    similarity_matrix = normalized_embeddings["image"] @ normalized_embeddings["text"].mT   # [image x text]

    cumulative_n_captions: torch.Tensor = torch.cumsum(torch.tensor([0] + n_captions), dim=0)
    x = torch.arange(cumulative_n_captions[-1])
    mask: torch.Tensor = (cumulative_n_captions[:-1, None] <= x) * (x < cumulative_n_captions[1:, None])

    return (similarity_matrix if return_similarity_matrix else None), TensorDict({
        "Image-to-Text": get_retrieval(similarity_matrix, mask),
        "Text-to-Image": get_retrieval(similarity_matrix.mT, mask.mT),
    }, batch_size=())


def print_retrieval_metrics(retrieval_metrics: TensorDict[str, TensorDict], indent: int = 0) -> None:
    # Print results    
    t = "\t" * indent
    for retrieval_type, d in retrieval_metrics.items():
        print(f"{t}{retrieval_type} Retrieval Metrics:")
        for k, value in utils.sort_dict(d).items():
            print(f"{t}{k}: {(100 * value.item()):.2f}%")
