from collections import OrderedDict

import einops
import torch
import torch.nn.functional as Fn
from modeling.image_features import ImageFeatures
from visualize.base import VISUALIZED_INDICES



def massive_token_heuristic(layer_idx: int, per_metric_output_dict: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    log_norms = einops.rearrange(
        torch.norm(per_metric_output_dict["layer_output"][layer_idx][0][:, ImageFeatures.image_indices, :], p=2, dim=-1).log(),
        "bsz (h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
    )
    flattened_norms = torch.sort(torch.flatten(log_norms), dim=0).values
    cutoff = torch.argmax(torch.diff(flattened_norms, dim=0), dim=0)
    mask = log_norms > flattened_norms[cutoff]
    return mask


def mask_attention_sink(
    attention: torch.Tensor,
    masked_tokens: torch.Tensor = None,
    max_num_tokens: int = None,
    scale: float = 1.0,
    verbose: bool = False
) -> torch.Tensor:
    bsz = attention.shape[0]
    if masked_tokens is None:
        masked_tokens = torch.full((bsz, ImageFeatures.N + 1), False)
    else:
        masked_tokens = masked_tokens.clone()
    masked_tokens[:, 0] = True
    
    attention = attention.clone()
    if attention.ndim == 4:
        attention = torch.mean(attention, dim=-1)                               # [bsz x n x n]

    method = "cls"
    if method in ["raw", "std"]:
        attention[:, :, 0] = 0.0
        attention = Fn.normalize(attention, p=1, dim=-1)
        attention[:, torch.arange(ImageFeatures.N + 1), torch.arange(ImageFeatures.N + 1)] = 0.0
        incoming_attention = torch.mean(attention, dim=1)                           # [bsz x n]
        
        values, indices = torch.max(incoming_attention, dim=1)                  # [bsz], [bsz]
        mask = torch.full_like(incoming_attention, False, dtype=torch.bool)     # [bsz x n]
        
        if method == "raw":
            scale = 9.0
            threshold = scale / (ImageFeatures.N + 1 - torch.sum(masked_tokens, dim=1))
        
        elif method == "std":
            scale = 7.0
            
            unmasked_tokens = ~masked_tokens
            count = torch.sum(unmasked_tokens, dim=1)
            
            use_log = False
            if use_log:
                _incoming_attention = torch.log(incoming_attention).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            else:
                _incoming_attention = incoming_attention
            
            mean = torch.sum(_incoming_attention * unmasked_tokens, dim=1) / count   # [bsz]
            std = (torch.sum(((_incoming_attention - mean[:, None]) ** 2) * unmasked_tokens, dim=1) / (count - 1)) ** 0.5    # [bsz]
            threshold = mean + scale * std
            
            if use_log:
                threshold = torch.exp(threshold)
        
        mask[range(bsz), indices] = (values > threshold)
        
        if verbose:
            for image_idx in VISUALIZED_INDICES:
                _attn = incoming_attention[image_idx][~masked_tokens[image_idx]]
                if use_log:
                    _attn = _attn.log()
                _sorted_attn = torch.sort(_attn, dim=0, descending=True).values
                
                print(f"\tImage {image_idx} --- incoming attention: {_sorted_attn[:10].tolist()}")
                print(f"\tnormalized incoming attention: {((_sorted_attn[:10] - torch.mean(_attn)) / torch.std(_attn)).tolist()}")
                
            print(f"Threshold: {threshold[VISUALIZED_INDICES].squeeze().tolist()}")
            
    elif method == "cls":
        attention = attention[:, 0, :]
        attention[:, 1:] = attention[:, 1:] / scale
        
        
        if max_num_tokens is None:
            mask = (attention > attention[:, :1])
        else:
            values, indices = torch.topk(attention[:, 1:], k=max_num_tokens, dim=1)    # [bsz]
            mask = torch.full((bsz, ImageFeatures.N + 1), False)
            mask[torch.arange(bsz)[:, None], indices + 1] = (values > attention[:, :1])
        
        if verbose:
            for image_idx in VISUALIZED_INDICES:
                _attn = attention[image_idx]
                print(f"\tImage {image_idx} --- CLS attention: {_attn[0].item()}, {(scale * torch.topk(_attn[1:], k=5).values).tolist()}")
    
    else:
        raise ValueError(method)
        
    mask = mask * ~masked_tokens
    return mask

