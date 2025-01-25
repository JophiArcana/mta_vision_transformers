import collections
import itertools
import operator
from typing import List, OrderedDict, Sequence, Union

import einops
import tensordict
import torch
from tensordict import TensorDict

from infrastructure import utils


class ImageFeatures(object):
    H = W = 16
    N = H * W
    image_indices = torch.arange(1, N + 1)

    ALL = "all"
    VALID = "valid"
    IMAGE = "image"
    CLS = "cls"

    keys = (ALL, VALID, IMAGE, CLS,)

    @classmethod
    def process_key(cls, key: int):
        return f"layer{key}"

    def __init__(
        self,
        per_layer_features: List[TensorDict],       # L x [B x ~N x ?]
        mta_masks: OrderedDict[int, torch.Tensor],  # K x [B x H x W]
    ):
        self.num_layers = len(per_layer_features)
        self.device = (*mta_masks.values(),)[0].device
        
        with utils.default_device(self.device):
            mta_lengths: List[int] = [
                torch.max(torch.sum(mta_mask, dim=(1, 2))).item()
                for mta_mask in mta_masks.values()
            ]
            mta_cutoffs: List[int] = [*itertools.accumulate((ImageFeatures.N + 1, *mta_lengths), operator.add)]

            # Stack per-layer features by padding to equal number of tokens with NaNs
            token_dim = 1
            padded_per_layer_features, d = [], max(features.shape[token_dim] for features in per_layer_features)
            assert d == ImageFeatures.N + 1 + sum(mta_lengths)

            for features in per_layer_features:
                pad_size = [0] * (2 * token_dim + 1) + [d - features.shape[token_dim]]
                padded_per_layer_features.append(tensordict.pad(features, pad_size, value=torch.nan))
            self.features: TensorDict = TensorDict.maybe_dense_stack(padded_per_layer_features, dim=0)      # [L x B x ~N x ?]
            self.shape = self.features.shape[:3]                                                            # (L, B, ~N)

            # Construct mask dict
            self.masks: OrderedDict[str, Union[torch.Tensor, TensorDict]] = collections.OrderedDict([
                (ImageFeatures.ALL, torch.tensor(True)),                                                    # []
                (ImageFeatures.VALID, TensorDict.apply(
                    TensorDict.isfinite(self.features),                                                     # [L x B x ~N x ?]
                    lambda t: torch.all(t, dim=-1),
                ))                                                                                          # [L x B x ~N]
            ])

            # Construct image mask
            image_mask = torch.full(self.shape[-1:], False)                                                 # [~N]
            image_mask[ImageFeatures.image_indices] = True
            self.masks[ImageFeatures.IMAGE] = image_mask

            # Construct cls mask
            cls_mask = torch.full(self.shape[-1:], False)                                                   # [~N]
            cls_mask[0] = True
            self.masks[ImageFeatures.CLS] = cls_mask

            # Construct MA masks
            for (start, end), (layer_idx, mta_mask) in zip(itertools.pairwise(mta_cutoffs), mta_masks.items()):
                mask = torch.full(self.shape, False)                                                        # [L x B x ~N]

                mask[:layer_idx + 1, :, ImageFeatures.image_indices] = einops.rearrange(mta_mask, "b h w -> b (h w)")
                mask[layer_idx + 1:, :, start:end] = (torch.arange(end - start) < torch.sum(mta_mask, dim=(1, 2))[:, None])
                self.masks[ImageFeatures.process_key(layer_idx)] = mask

            # Expand tensor masks to tensordicts
            for k, mask in self.masks.items():
                if isinstance(mask, torch.Tensor):
                    self.masks[k] = self._expand_mask_to_tensordict(mask)

    def _expand_mask_to_tensordict(self, mask: torch.Tensor) -> TensorDict:
        expanded_mask = mask.expand(self.shape)
        return TensorDict.apply(self.features, lambda _: expanded_mask)

    def _accumulate(self, queries: Sequence[str | int]) -> TensorDict:
        with utils.default_device(self.device):
            mask = self._expand_mask_to_tensordict(torch.tensor(False))
            for query in queries:
                if isinstance(query, int):
                    query = ImageFeatures.process_key(query)
                mask = mask + self.masks.get(query, torch.tensor(False))
            return mask

    def get(
        self,
        layer_idx: int = None,
        key: str = None,
        include: Sequence[str | int] = (ALL,),
        exclude: Sequence[str | int] = (),
        require_valid: bool = True,
    ) -> Union[torch.Tensor, OrderedDict[str, torch.Tensor]]:
        with utils.default_device(self.device):
            include_mask = self._accumulate(include)
            exclude_mask = self._accumulate(exclude)

            mask = include_mask * TensorDict.apply(exclude_mask, torch.Tensor.logical_not)
            if require_valid:
                mask = mask * self.masks[ImageFeatures.VALID]

            if key is None:
                keys = self.features.keys(include_nested=True, leaves_only=True)
            else:
                keys = (key,)

            if layer_idx is None:
                result = collections.OrderedDict([
                    (k, einops.rearrange(
                        self.features[k][mask[k].expand(self.shape)],
                        "(l t) d -> l t d", l=self.num_layers
                    )) for k in keys
                ])
            else:
                result = collections.OrderedDict([
                    (k, self.features[k][layer_idx, mask[k].expand(self.shape)[layer_idx]])
                    for k in keys
                ])

            if key is None:
                return result
            else:
                return result[key]

