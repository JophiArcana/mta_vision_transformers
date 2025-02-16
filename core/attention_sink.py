from collections import OrderedDict

import einops
import torch
import torch.nn.functional as Fn
from core.visualize import num_visualized_images
from modeling.image_features import ImageFeatures



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
    verbose: bool = False
) -> torch.Tensor:
    bsz = attention.shape[0]
    
    if masked_tokens is None:
        masked_tokens = torch.full((bsz, ImageFeatures.N + 1), False)
    else:
        masked_tokens = masked_tokens.clone()
    masked_tokens[:, 0] = True
    
    attention = torch.mean(attention, dim=-1)                                   # [bsz x n x n]
    attention[:, :, 0] = 0.0
    attention = Fn.normalize(attention, p=1, dim=-1)
    attention[:, range(ImageFeatures.N + 1), range(ImageFeatures.N + 1)] = 0.0
    incoming_attention = torch.mean(attention, dim=1)                           # [bsz x n]
    

    values, indices = torch.max(incoming_attention, dim=1)                  # [bsz], [bsz]
    mask = torch.full_like(incoming_attention, False, dtype=torch.bool)     # [bsz x n]


    method = "std"
    if method == "raw":
        scale = 9.0
        threshold = scale / (ImageFeatures.N + 1 - torch.sum(masked_tokens, dim=1))
    
    elif method == "std":
        scale = 3.0
        
        unmasked_tokens = ~masked_tokens
        count = torch.sum(unmasked_tokens, dim=1)
        
        log_incoming_attention = torch.log(incoming_attention).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        mean = torch.sum(log_incoming_attention * unmasked_tokens, dim=1) / count   # [bsz]
        std = (torch.sum(((log_incoming_attention - mean[:, None]) ** 2) * unmasked_tokens, dim=1) / (count - 1)) ** 0.5    # [bsz]
        
        threshold = torch.exp(mean + scale * std)
    
    else:
        raise ValueError(method)
    
    mask[range(bsz), indices] = (values > threshold)
    
    
    # mask = incoming_attention > threshold
    
    # sorted_attention, indices = torch.sort(incoming_attention[:, ImageFeatures.image_indices], dim=1, descending=True)
    # cutoff_idx = torch.argmax(torch.diff(sort))
    
    mask = mask * ~masked_tokens
    
    if verbose:
        for image_idx in range(num_visualized_images):
            _attn = incoming_attention[image_idx][~masked_tokens[image_idx]].log()
            _sorted_attn = torch.sort(_attn, dim=0, descending=True).values
            
            print(f"\tImage {image_idx} --- incoming attention: {_sorted_attn[:10].tolist()}")
            print(f"\tnormalized incoming attention: {((_sorted_attn[:10] - torch.mean(_attn)) / torch.std(_attn)).tolist()}")
            
        print(f"Threshold: {threshold[:num_visualized_images].squeeze().tolist()}")

    return mask

