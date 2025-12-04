import collections
import itertools
import operator
from typing import List, OrderedDict, Sequence, Tuple, Union

import einops
import torch
import torch.nn.functional as Fn
from tensordict import TensorDict

from infrastructure import utils


class ImageFeatures(object):
    H = W = 16
    N = H * W
    NUM_LAYERS = 24
    image_indices = [*range(1, N + 1)]

    ALL = "all"
    VALID = "valid"
    IMAGE = "image"
    CLS = "cls"

    keys = (ALL, VALID, IMAGE, CLS,)

    @classmethod
    def process_key(cls, key: Union[str, int]):
        return f"layer{key}" if isinstance(key, int) else key

    @classmethod
    def from_tensor(
        cls,
        t: torch.Tensor,                            # [B x ~N x ?]
        mta_masks: OrderedDict[int, torch.Tensor],  # K x [B x H x W]
        mode: str,
        output_device: str,
    ) -> "ImageFeatures":
        return ImageFeatures(
            [TensorDict({"": t}, batch_size=t.shape[:2]).auto_device_()],
            mta_masks, mode, output_device,
        )

    def __init__(
        self,
        per_layer_features: List[TensorDict],       # L x [B x ~N x ?]
        mta_masks: OrderedDict[int, torch.Tensor],  # K x [B x H x W]
        mode: str,
        output_device: str,
    ):
        valid_layers = [idx for idx, f in enumerate(per_layer_features) if f is not None]    
        self.num_layers = len(valid_layers)
        self.layer_map = OrderedDict(zip(valid_layers, range(self.num_layers)))
        per_layer_features = [per_layer_features[idx] for idx in valid_layers]
        
        self.mode = mode
        self.device = per_layer_features[0].device
        self.output_device = output_device
        
        with utils.default_device(self.device):
            mta_lengths: List[int] = [
                torch.max(torch.sum(mta_mask, dim=1)).item()
                for mta_mask in mta_masks.values()
            ]
            mta_cutoffs: List[int] = [*itertools.accumulate((ImageFeatures.N + 1, *mta_lengths), operator.add)]

            # Stack per-layer features by padding to equal number of tokens with NaNs
            padded_per_layer_features = [TensorDict() for _ in range(self.num_layers)]
            for k in per_layer_features[0].keys(include_nested=True, leaves_only=True):
                per_layer_values = [features[k] for features in per_layer_features]
                shape = torch.max(torch.stack([
                    torch.tensor(values.shape)
                    for values in per_layer_values
                ], dim=0), dim=0).values
                
                for padded_features, values in zip(padded_per_layer_features, per_layer_values):
                    pad_size = (*itertools.chain(*(
                        (0, shape[i] - values.shape[i])
                        for i in reversed(range(len(shape)))
                    )),)
                    padded_features[k] = Fn.pad(values, pad_size, mode="constant", value=torch.nan)
            self.features: TensorDict = TensorDict.maybe_dense_stack(padded_per_layer_features, dim=0)
            self.features = self.features.auto_batch_size_(batch_dims=3)    # [L x B x ~N x ?]
            self.shape = self.features.shape[:3]                                                            # (L, B, ~N)

            # Construct mask dict
            self.masks: OrderedDict[str, Union[torch.Tensor, TensorDict]] = collections.OrderedDict([
                (ImageFeatures.ALL, torch.tensor(True)),                                                    # []
                (ImageFeatures.VALID, TensorDict.apply(
                    TensorDict.isfinite(self.features),                                                     # [L x B x ~N x ?]
                    lambda t: torch.all(t.flatten(3, -1), dim=-1),
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

                if self.mode in ["concatenation"]:
                    mask[:layer_idx + 1, :, :ImageFeatures.N + 1] = mta_mask
                    mask[layer_idx + 1:, :, start:end] = (torch.arange(end - start) < torch.sum(mta_mask, dim=1)[:, None])
                elif self.mode in ["mean_concatenation"]:
                    mask[:layer_idx + 1, :, :ImageFeatures.N + 1] = mta_mask
                    mask[layer_idx + 1:, :, start] = True
                # elif self.mode in ["default", "mean_substitution", "permutation"]:
                else:
                    mask[:, :, :ImageFeatures.N + 1] = mta_mask
        
                self.masks[ImageFeatures.process_key(layer_idx)] = mask

    def update(self, key: Union[str, Tuple[str, ...]], value: Union[torch.Tensor, TensorDict]) -> None:
        self.features[key] = value
        self.masks[ImageFeatures.VALID][key] = torch.all(torch.isfinite(value).flatten(3, -1), dim=-1)

    def _expand_mask_to_tensordict(self, mask: Union[torch.Tensor, TensorDict]) -> TensorDict:
        if isinstance(mask, torch.Tensor):
            expanded_mask = mask.expand(self.shape)
            return TensorDict.apply(self.features, lambda _: expanded_mask)
        else:
            return mask

    def _accumulate(self, queries: Sequence[str | int]) -> TensorDict:
        with utils.default_device(self.device):
            mask = self._expand_mask_to_tensordict(torch.tensor(False))
            for query in queries:
                if isinstance(query, int):
                    query = ImageFeatures.process_key(query)
                mask = mask + self._expand_mask_to_tensordict(self.masks.get(query, torch.tensor(False)))
            return mask

    def get(
        self,
        layer_idx: int = None,
        key: str = None,
        include: Sequence[str | int] = (ALL,),
        exclude: Sequence[str | int] = (),
        with_batch: bool = False,
        require_valid: bool = True,
    ) -> Union[torch.Tensor, OrderedDict[str, torch.Tensor]]:
        if layer_idx is not None:
            if layer_idx not in self.layer_map:
                return None
            else:
                layer_idx = self.layer_map[layer_idx]
        
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
                if with_batch:
                    pattern, dims = "(l bsz t) ... -> l bsz t ...", {"l": self.num_layers, "bsz": self.shape[1]}
                else:
                    pattern, dims = "(l t) ... -> l t ...", {"l": self.num_layers}
                
                result = collections.OrderedDict([
                    (k, einops.rearrange(
                        self.features[k][mask[k].expand(self.shape)],
                        pattern, **dims,
                    ).to(self.output_device)) for k in keys
                ])
            else:
                if with_batch:
                    pattern, dims = "(bsz t) ... -> bsz t ...", {"bsz": self.shape[1]}
                else:
                    pattern, dims = "t ... -> t ...", {}
                
                result = collections.OrderedDict([
                    (k, einops.rearrange(
                        self.features[k][layer_idx, mask[k].expand(self.shape)[layer_idx]],
                        pattern, **dims,
                    ).to(self.output_device)) for k in keys
                ])

            if key is None:
                return result
            else:
                return result[key]

