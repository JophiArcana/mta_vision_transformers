import os
import sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("/workspace/mta_vision_transformers/")

import torch
import torch.nn as nn
from diffusers import StableDiffusion3Pipeline

from modeling.vit_nystrom import StableDiffusion3NystromCompressionViT


if __name__ == "__main__":
    torch.manual_seed(12122002)
    torch.set_printoptions(linewidth=400, sci_mode=False)
    
    start_layer, end_layer = 12, 24
    def layer_condition(module: nn.Module, name: str) -> bool:
        mask_layers = [*range(start_layer, end_layer)]
        return name in [
            f"model.transformer.transformer_blocks.{idx}.attn" for idx in mask_layers
        ] + [
            f"model.transformer.transformer_blocks.{idx}.attn2" for idx in mask_layers
        ]
    start_time, end_time = 6, 18
    def timestep_condition(t: int) -> bool:
        return start_time <= t < end_time
    
    # model = StableDiffusion3Pipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-3.5-medium",
    #     torch_dtype=torch.bfloat16,
    # ).to("cuda")
    model = StableDiffusion3NystromCompressionViT(
        "fps",
        compression_mode="linear",
        num_sample=[256, 256],
        resample=False,
        use_layer_input=True,
        layer_condition=layer_condition,
        timestep_condition=timestep_condition,
    )

    prompts = [
        "A capybara holding a sign that reads Hello World",
        # "A fat cat sleeping next to a golden retriever",
    ]
    images = model(
        prompts,
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images
    model._cache.clear()

    
    images[0].save("capybara1212.png")
    # images[1].save("Miyu and Joe.png")
