import types
from typing import Callable, Dict, List, Literal, Optional, Tuple

import einops
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.model import CLIP
from open_clip.transformer import ResidualAttentionBlock

from infrastructure.settings import SEED, DEVICE



ModeOptions = Literal[
    "default",
    "concatenation",
    "mean_substitution",
    "permutation",
    "mean_concatenation",
    "permutation_concatenation",
]
class OpenCLIPViT(nn.Module):
    @classmethod
    def return_module_name(cls, handle: str) -> str:
        return f"return_{handle}"
    
    def __init__(
        self,
        cutoff: Dict[str, Dict[str, Callable[[torch.Tensor], torch.Tensor]]],
        mode: ModeOptions = "concatenate",
    ):
        super().__init__()
        model: CLIP = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", force_quick_gelu=True)[0]
        
        
        
        
        
        
        def new_resblock_attention(
            self: ResidualAttentionBlock,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None, 
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None
        ):
            assert k_x is None and v_x is None, "Only implemented for k_x and v_x as None"
            attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
            
            qkv = F.linear(q_x, self.attn.in_proj_weight, self.attn.in_proj_bias)
            q_x, k_x, v_x = einops.rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.attn.num_heads)
            
            attn_weights = torch.matmul(q_x, k_x.mT) / (q_x.shape[-1] ** 0.5)
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            attn_matrix = F.softmax(attn_weights, dim=-1)
            
            x = einops.rearrange(torch.matmul(attn_matrix, v_x), "b h n d -> b n (h d)")
            x = F.linear(x, self.attn.out_proj.weight, self.attn.out_proj.bias)
                
            for k in self.attention_returns:
                self.get_submodule(OpenCLIPViT.return_module_name(k))(locals()[k])
            return x
        
        for idx, blk in enumerate(model.visual.transformer.resblocks):
            blk: ResidualAttentionBlock
            blk.attention_returns: List[str] = ["attn_matrix",]
            for handle in blk.attention_returns:
                blk.register_module(OpenCLIPViT.return_module_name(handle), nn.Identity())
            blk.attention = types.MethodType(new_resblock_attention, blk)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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
                    
                    mta_mask = projection_condition(projections)
                    
                    if mode in ["concatenation", "mean_concatenation", "permutation_concatenation"]:
                        # Add top-k tokens as register tokens after cls
                        if mode in ["concatenation", "permutation_concatenation"]:
                            r = torch.sum(mta_mask, dim=-1)
                            max_r = torch.max(r).item()
                            
                            register_tokens = torch.full((bsz, max_r, D), 0.0)
                            
                            register_mask = torch.arange(max_r) < r[:, None]    # [batch_size, max_r]
                            if mode == "concatenation":
                                register_tokens[register_mask] = image_x[mta_mask]
                            elif mode == "permutation_concatenation":
                                register_tokens[register_mask] = image_x[mta_mask][torch.randperm(torch.sum(r).item())]
                            
                            _attn_mask = F.pad(_attn_mask, (0, max_r, 0, max_r), mode="constant", value=0.0)
                            for im_idx, im_r in enumerate(r):
                                if im_r != max_r:
                                    _attn_mask[im_idx, :, im_r - max_r:] = -torch.inf
                                
                        elif mode == "mean_concatenation":
                            mta_indices = torch.unbind(torch.argwhere(mta_mask) + torch.tensor([0, 1]), dim=-1)
                            register_tokens = torch.mean(x[mta_indices], dim=0).expand((bsz, 1, D))
                        
                            _attn_mask = F.pad(_attn_mask, (0, 1, 0, 1), mode="constant", value=0.0)                        
                        
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

                    attn_mask = _attn_mask[:, None, :, :]   # torch.repeat_interleave(_attn_mask, n_heads, dim=0)
                    mta_masks[idx] = torch.cat((torch.full((bsz, 1), False), mta_mask.flatten(1, 2)), dim=1)
            
            if not self.batch_first:
                x = x.transpose(0, 1)    # LND -> NLD
                
            return x
        
        model.visual.transformer.forward = types.MethodType(new_transformer_forward, model.visual.transformer)

        self.model = model
        self.model.eval()
        self.mta_masks = mta_masks
        
    def forward(self, x):
        return self.model(x), self.mta_masks
