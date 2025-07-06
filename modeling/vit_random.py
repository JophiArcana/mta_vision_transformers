import types
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple

import einops
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.model import CLIP
from open_clip.transformer import ResidualAttentionBlock, Transformer
from torch_pca import PCA

from infrastructure.settings import SEED, DEVICE
from modeling.image_features import ImageFeatures
from modeling.base_vit import OpenCLIPViT



class OpenCLIPRandomViT(OpenCLIPViT):
    ModeOptions = Literal[
        "random_input",
        "random_initialization",
        "permute_layers",
    ]
    
    def __init__(
        self,
        mode: Set[ModeOptions],
        layer_order: List[int] = None,
        random_input_layer: int = 0,
    ):
        nn.Module.__init__(self)
        self.mode: Set[OpenCLIPRandomViT.ModeOptions] = mode

        kwargs = OpenCLIPViT.INITIALIZE_KWARGS.copy()
        if "random_initialization" in self.mode:
            kwargs["pretrained"] = None
        self.model: CLIP = open_clip.create_model_and_transforms(**kwargs)[0]
        self.model.eval()
        
        self.layer_order: List[int] = layer_order
        self.random_input_layer: int = random_input_layer
        
        
        # SECTION: Replace transformer.forward
        def new_transformer_forward(_self: Transformer, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
            if self.layer_order is None:
                if "permute_layers" in self.mode:
                    self.layer_order = torch.randperm(_self.layers, dtype=torch.int).tolist()
                else:
                    self.layer_order = torch.arange(_self.layers, dtype=torch.int).tolist()
            
            self.random_input_layer = self.layer_order[torch.searchsorted(torch.tensor(self.layer_order), self.random_input_layer)]
            for idx in self.layer_order:
                if "random_input" in self.mode and idx == self.random_input_layer:        
                    pca = PCA(n_components=x.shape[0] // 2, whiten=True)
                
                    flattened_x = torch.flatten(x, start_dim=0, end_dim=-2)
                    transformed_x = pca.fit_transform(flattened_x)
                    transformed_x = F.normalize(torch.randn_like(transformed_x), p=2, dim=-1)
                    flattened_x = pca.inverse_transform(transformed_x)
            
                    x = flattened_x.view_as(x)
                    
                    # if True:
                    #     k = 4
                    #     p = torch.tensor([i * ImageFeatures.W + j for i in range(k) for j in range(k)]) + 1
                    #     # c = torch.mean(x[:, p], dim=1, keepdim=True)
                    #     c = x[:, 1:2]
                        
                    #     x[:, p] = c + 0.1 * (x[:, p] - c)
                
                x = _self.resblocks[idx](x, attn_mask=attn_mask)  
            return x
        
        self.model.visual.transformer.forward = types.MethodType(new_transformer_forward, self.model.visual.transformer)
