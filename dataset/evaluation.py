import os
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple

import datasets
import torch
import torch.nn.functional as Fn
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler, Sampler

from infrastructure import utils
from infrastructure.settings import DEVICE
from dataset.library import DATASETS
from modeling.openclip_vit import OpenCLIPViT


# COCODataset class with text captions
class ImageTextDataset(Dataset):
    mode_transforms: Dict[str, Callable[[Any], torch.Tensor]] = {
        "image": OpenCLIPViT.preprocess_func,
        "text": OpenCLIPViT.tokenizer_func,
    }
    
    def __init__(self,
                 dataset_name: str,
                 keys: Dict[str, str],
                 split: str = "train",
                 cache_fnames: Dict[str, str] = {},
    ):
        self.data = datasets.load_dataset(dataset_name, split=split, trust_remote_code=True)
        self.keys = keys
        
        self.cache: Dict[str, torch.Tensor] = {}
        self._reset_cache(cache_fnames)

    def _reset_cache(self, cache_fnames: Dict[str, str]) -> None:
        self.cache = {}
        for k, fname in cache_fnames.items():
            if os.path.exists(fname):
                self.cache[k] = torch.load(fname, map_location=DEVICE)

    def __len__(self) -> int:
        return self.data.num_rows

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        return ({
            mode: transform(self.data[idx][self.keys[mode]]).to(DEVICE)
            for mode, transform in ImageTextDataset.mode_transforms.items()
        }, {k: mask[idx] for k, mask in self.cache.items()})


DEFAULT_DATASET = ImageTextDataset(DATASETS["Common"][1][0], keys={"image": "image", "text": "caption"}, split="train")
def run_retrieval_evaluation(
    model: OpenCLIPViT,
    dataset: Dataset = DEFAULT_DATASET,
    subsample: int = None,
    k_values: List[int] = [1, 2, 5],
) -> Dict[str, Dict[str, float]]:
    """
    Compute retrieval metrics for the validation dataset.
    
    Args:
    - model: OpenCLIP model
    - dataset: Validation DataLoader
    - device: Computation device
    
    Returns:
    - dict: Retrieval metrics
    """
    # Collect image and text embeddings
    model.eval()
    if subsample is not None:
        dataset = Subset(dataset, torch.randperm(len(dataset))[:subsample].tolist())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, generator=torch.Generator(DEVICE))
    model.model.visual.forward
    embedding_lists: Dict[str, List[torch.Tensor]] = {}
    
    forward_cache: Dict[str, torch.Tensor] = getattr(model.__class__, "_cache", {})
    with torch.no_grad():
        for data, cached_data in tqdm(dataloader):
            forward_cache.update(cached_data)
            
            data: Dict[str, torch.Tensor]
            for k, t in data.items():
                embedding_lists.setdefault(k, []).append(getattr(model.model, f"encode_{k}")(t, normalize=True))
                
            forward_cache.clear()
            utils.empty_cache()
            

    # Concatenate embeddings
    normalized_embeddings: Dict[str, torch.Tensor] = {k: torch.cat(v, dim=0) for k, v in embedding_lists.items()}
    similarity_matrix = normalized_embeddings["image"] @ normalized_embeddings["text"].mT   # [N x M]

    def get_retrieval(S: torch.Tensor) -> Dict[str, float]:             # [N x M]
        rank = torch.sum(S > torch.diag(S)[:, None], dim=1)             # [N]
        return {
            f"R@{k}": torch.mean((rank < k).to(torch.float)).item()
            for k in k_values
        }

    return {
        "text_to_image": get_retrieval(similarity_matrix),
        "image_to_text": get_retrieval(similarity_matrix.mT),
    }


def print_retrieval_metrics(retrieval_metrics: Dict[str, Dict[str, float]], indent: int = 0) -> None:
    # Print results
    t = "\t" * indent
    print(f"{t}Text-to-Image Retrieval Metrics:")
    for k, value in retrieval_metrics["text_to_image"].items():
        print(f"{t}{k}: {(100 * value):.2f}%")

    print(f"{t}Image-to-Text Retrieval Metrics:")
    for k, value in retrieval_metrics["image_to_text"].items():
        print(f"{t}{k}: {(100 * value):.2f}%")
