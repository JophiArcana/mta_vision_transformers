import types
from typing import Callable, Dict, Literal, Optional, Tuple

import einops
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from infrastructure.settings import SEED, DEVICE



ModeOptions = Literal["default", "concatenation", "mean_substitution", "permutation"]
class OpenCLIPViT(nn.Module):
    def __init__(
        self,
        cutoff: Dict[str, Dict[str, Callable[[torch.Tensor], torch.Tensor]]],
        mode: ModeOptions = "concatenate",
    ):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", force_quick_gelu=True)
        
        mta_masks = {}
        def new_transformer_forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):            
            bsz, _, D = x.shape
            N = 256
            H = W = int(N ** 0.5)
            head_dim = 64
            n_heads = D // head_dim
            
            _attn_mask = torch.zeros((bsz, N + 1, N + 1))
            
            if not self.batch_first:
                x = x.transpose(0, 1).contiguous()    # NLD -> LND
                
            def get_image_x(x_: torch.Tensor) -> torch.Tensor:
                return x_[:, 1:N + 1]

            for idx, r in enumerate(self.resblocks):
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
                
                if idx in cutoff:
                    torch.manual_seed(SEED)
                    np.random.seed(SEED)
                    
                    cutoff_dict = cutoff[idx]
                    projection_condition = cutoff_dict["condition"]

                    ############################################################################################################
                    # Select top-k tokens by norm
                    image_x = transformed_image_x = get_image_x(x)
                    if "pre_hook" in cutoff_dict:
                        transformed_image_x = cutoff_dict["pre_hook"](transformed_image_x)
                    
                    demeaned_image_x = transformed_image_x - torch.mean(transformed_image_x.flatten(0, -2), dim=0)
                    V = torch.linalg.svd(demeaned_image_x.flatten(0, -2), full_matrices=False)[-1].mT             # [hidden_size]
                    projections = demeaned_image_x @ V  
                                        
                    if "post_hook" in cutoff_dict:
                        projections = cutoff_dict["post_hook"](projections)
                    else:
                        projections = projections[..., 0]
                    
                    from matplotlib import pyplot as plt
                    plt.hist(projections.flatten().numpy(force=True), bins=1000)
                    plt.show()
                    
                    # plt.imshow(projections[6].reshape(16, 16).numpy(force=True))
                    # plt.show()
                    
                    mta_mask = projection_condition(projections)
                    
                    if mode == "concatenation":
                        # Add top-k tokens as register tokens after cls
                        r = torch.sum(mta_mask, dim=-1)
                        max_r = torch.max(r).item()
                        
                        register_tokens = torch.full((bsz, max_r, D), 0.0)
                        
                        _attn_mask = F.pad(_attn_mask, (0, max_r, 0, max_r), mode="constant", value=0.0)
                        for im_idx, im_r in enumerate(r):
                            register_tokens[im_idx, :im_r] = image_x[im_idx][mta_mask[im_idx]]
                            if im_r != max_r:
                                _attn_mask[im_idx, :, im_r - max_r:] = -torch.inf
                        
                        mta_mask = einops.rearrange(mta_mask, "b (h w) -> b h w", h=H, w=W)
                        
                        # Reshape image tokens to grid
                        image_x_grid = einops.rearrange(image_x, "b (h w) c -> b c h w", h=H, w=W)  # [batch_size, hidden_size, h, w]

                        # Pad the tokens and mask to handle 3x3 neighborhoods at boundaries
                        nb = 7
                        p = nb // 2
                        padded_image_x_grid = F.pad(image_x_grid, (p, p, p, p), mode="reflect")
                        padded_mask = F.pad(mta_mask, (p, p, p, p), mode="constant", value=False)

                        # Extract 3x3 neighborhoods
                        neighborhood = torch.stack([
                            padded_image_x_grid[:, :, i:i + H, j:j + W]
                            for i in range(nb) for j in range(nb)
                        ], dim=-1)  # [batch_size, hidden_size, height, width, 9]
                        neighborhood_mask = torch.stack([
                            padded_mask[:, i:i + H, j:j + W]
                            for i in range(nb) for j in range(nb)
                        ], dim=-1).unsqueeze(1) # [batch_size, 1, height, width, 9]

                        # Mask invalid neighbors (masked tokens set to large negative values instead of inf)
                        valid_neighbors = torch.where(~neighborhood_mask, neighborhood, -torch.inf)  # [batch_size, hidden_size, height, width, 9]

                        # Compute the norm of valid_neighbors across the hidden_size
                        neighbor_norms = valid_neighbors.norm(dim=1)  # Norm across hidden_size: [batch_size, height, width, 9]

                        # Mask out invalid norms (set them to -inf so they don't participate in sorting/median selection)
                        neighbor_norms_masked = torch.where(~neighborhood_mask.squeeze(1), neighbor_norms, -torch.inf)

                        # Identify the median norm neighbor for each token
                        sorted_norms, sorted_indices = neighbor_norms_masked.sort(dim=-1, descending=True)  # Sort descending to exclude -inf
                        valid_sorted = sorted_norms > float('-inf')  # Track valid neighbors
                        median_index = valid_sorted.sum(dim=-1) // 2  # Find the median index among valid neighbors
                        median_indices = sorted_indices.gather(dim=-1, index=median_index.unsqueeze(-1)).squeeze(-1)  # [batch_size, height, width]

                        # Expand median_indices to match valid_neighbors dimensions
                        median_indices_expanded = median_indices[:, None, ..., None].expand(-1, D, -1, -1, 1)

                        # Gather tokens corresponding to the median norm
                        median_neighbors = torch.gather(valid_neighbors, dim=-1, index=median_indices_expanded).squeeze(-1)  # [batch_size, hidden_size, height, width]

                        # Replace top-k tokens with their median neighbors
                        image_x_grid_filled = torch.where(mta_mask[:, None, :, :], median_neighbors, image_x_grid)

                        # Reshape back to sequence format
                        image_x_filled = einops.rearrange(image_x_grid_filled, "b c h w -> b (h w) c")
                        x = torch.cat((x[:, :1], image_x_filled, x[:, N + 1:], register_tokens), dim=1)
                    
                    else:
                        mta_indices = torch.unbind(torch.argwhere(mta_mask) + torch.tensor([0, 1]), dim=-1)
                        if mode == "mean_substitution":
                            x[mta_indices] = torch.mean(x[mta_indices], dim=0)
                            
                        elif mode == "permutation":
                            torch.seed()
                            x[mta_indices] = x[mta_indices][torch.randperm(len(mta_indices[0]), device=DEVICE)]

                        mta_mask = einops.rearrange(mta_mask, "b (h w) -> b h w", h=H, w=W)                    
                    ############################################################################################################

                    attn_mask = torch.repeat_interleave(_attn_mask, n_heads, dim=0)
                    mta_masks[idx] = mta_mask
            
            if not self.batch_first:
                x = x.transpose(0, 1)    # LND -> NLD
                
            return x
        
        model.visual.transformer.forward = types.MethodType(new_transformer_forward, model.visual.transformer)

        self.model = model
        self.model.eval()
        self.mta_masks = mta_masks
        
    def forward(self, x):
        return self.model(x), self.mta_masks
