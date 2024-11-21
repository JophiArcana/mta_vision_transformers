import types
from typing import Optional

import einops
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenCLIPViT(nn.Module):
    def __init__(self, version='ViT-L-14', pretrained='openai'):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(version, pretrained=pretrained)

        def new_transformer_forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
            ma_layer = 15
            if not self.batch_first:
                x = x.transpose(0, 1).contiguous()    # NLD -> LND

            for r in self.resblocks[:ma_layer]:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)

            ############################################################################################################
            layer_output = x  #[batch, sequence_length, hidden_size]

            # Select top-k tokens by norm
            k = 3
            top_k_tokens = torch.topk(layer_output[:, 1:, :].norm(dim=-1), k=k, dim=-1)  # [batch_size, k]
            self.top_k_indices = top_k_tokens.indices

            # Add top-k tokens as register tokens after cls
            layer_output_top_k = torch.gather(layer_output, 1, top_k_tokens.indices.unsqueeze(-1).expand(-1, -1, layer_output.shape[-1]) + 1)
            layer_output = torch.cat([layer_output[:, :1, :], layer_output_top_k, layer_output[:, 1:, :]], dim=1)

            # Reshape image tokens to grid
            hw = int(layer_output[:, 1 + k:, :].shape[1] ** 0.5)
            layer_output_image_tokens = einops.rearrange(layer_output[:, 1 + k:, :], "b (h w) c -> b c h w", h=hw)  # [batch_size, hidden_size, h, w]

            # Identify top-k token positions (row, col)
            rows, cols = top_k_tokens.indices // hw, top_k_tokens.indices % hw
            batch_size, hidden_size, height, width = layer_output_image_tokens.shape

            # Create a mask for the top-k tokens
            mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=layer_output_image_tokens.device)
            mask[torch.arange(batch_size).unsqueeze(-1), rows, cols] = True

            # Pad the tokens and mask to handle 3x3 neighborhoods at boundaries
            padded_image_tokens = F.pad(layer_output_image_tokens, (1, 1, 1, 1), mode='reflect')
            padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=False)

            # Extract 3x3 neighborhoods
            neighborhood = torch.stack([padded_image_tokens[:, :, i:i + height, j:j + width]
                                        for i in range(3) for j in range(3)], dim=-1)  # [batch_size, hidden_size, height, width, 9]
            neighborhood_mask = torch.stack([padded_mask[:, i:i + height, j:j + width]
                                            for i in range(3) for j in range(3)], dim=-1).unsqueeze(1)  # [batch_size, 1, height, width, 9]

            # Mask invalid neighbors (masked tokens set to large negative values instead of inf)
            valid_neighbors = torch.where(~neighborhood_mask, neighborhood, torch.tensor(float('-inf'), device=layer_output.device))  # [batch_size, hidden_size, height, width, 9]

            # Compute the norm of valid_neighbors across the hidden_size
            neighbor_norms = valid_neighbors.norm(dim=1)  # Norm across hidden_size: [batch_size, height, width, 9]

            # Mask out invalid norms (set them to -inf so they don't participate in sorting/median selection)
            neighbor_norms_masked = torch.where(~neighborhood_mask.squeeze(1), neighbor_norms, torch.tensor(float('-inf'), device=layer_output.device))

            # Identify the median norm neighbor for each token
            sorted_norms, sorted_indices = neighbor_norms_masked.sort(dim=-1, descending=True)  # Sort descending to exclude -inf
            valid_sorted = sorted_norms > float('-inf')  # Track valid neighbors
            median_index = valid_sorted.sum(dim=-1) // 2  # Find the median index among valid neighbors
            median_indices = sorted_indices.gather(dim=-1, index=median_index.unsqueeze(-1)).squeeze(-1)  # [batch_size, height, width]

            # Expand median_indices to match valid_neighbors dimensions
            median_indices_expanded = median_indices.unsqueeze(1).unsqueeze(-1).expand(-1, hidden_size, -1, -1, 1)

            # Gather tokens corresponding to the median norm
            median_neighbors = torch.gather(valid_neighbors, dim=-1, index=median_indices_expanded).squeeze(-1)  # [batch_size, hidden_size, height, width]

            # Replace top-k tokens with their median neighbors
            layer_output_image_tokens_filled = torch.where(mask.unsqueeze(1), median_neighbors, layer_output_image_tokens)

            # Reshape back to sequence format
            layer_output_image_tokens_filled = einops.rearrange(layer_output_image_tokens_filled, "b c h w -> b (h w) c")
            layer_output = torch.cat([layer_output[:, :1 + k, :], layer_output_image_tokens_filled], dim=1)

            x = layer_output
            ############################################################################################################
                    
            for r in self.resblocks[ma_layer:]:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)

            if not self.batch_first:
                x = x.transpose(0, 1)    # LND -> NLD
                
            return x
        
        model.visual.transformer.forward = types.MethodType(new_transformer_forward, model.visual.transformer)

        self.model = model
        self.model.eval()
        
    def forward(self, x):
        return self.model(x), self.model.visual.transformer.top_k_indices
