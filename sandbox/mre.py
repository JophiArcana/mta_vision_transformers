import os
import sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("/workspace/mta_vision_transformers/")

import torch
import torch.nn as nn

from modeling.vit_nystrom import StableDiffusion3NystromCompressionViT


if __name__ == "__main__":
    torch.set_printoptions(linewidth=400, sci_mode=False)
    
    def layer_condition(module: nn.Module, name: str) -> bool:
        mask_layers = [*range(12, 24)]
        return name in [
            f"model.transformer.transformer_blocks.{idx}.attn" for idx in mask_layers
        ] + [
            f"model.transformer.transformer_blocks.{idx}.attn2" for idx in mask_layers
        ]
    
    def timestep_condition(t: int) -> bool:
        return t < 14
    
    model = StableDiffusion3NystromCompressionViT(
        "fps",
        num_sample=[256, 256],
        resample=False,
        use_layer_input=True,
        layer_condition=layer_condition,
        timestep_condition=timestep_condition,
    )

    images = model(
        [
            "A capybara holding a sign that reads Hello World",
            "A fat cat sleeping next to a golden retriever",
        ],
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images
    images[0].save("capybara.png")
    images[1].save("Miyu and Joe.png")
