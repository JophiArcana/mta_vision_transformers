#%%
import os
import sys
sys.path.append("/workspace/mta_vision_transformers/")
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from matplotlib import pyplot as plt
from torch.utils._pytree import tree_flatten
from transformers import CLIPVisionModel

from infrastructure.settings import DEVICE, DTYPE
from dataset.construct import ImageDataset
from dataset.library import DATASETS

from core.monitor import Monitor
from core.visualize import visualize_model_outputs


if __name__ == "__main__":
    dataset_name, n_classes = DATASETS["Common"][1]
    
    # Organic: 1107349783
    torch.manual_seed(1212)
    seed = 1107349783   # np.random.randint(0, 1 << 31)
    np.random.seed(seed)
    print(f"Seed: {seed}")
    
    
    def residual_hook_fn(model_: nn.Module, input_: Any, output_: Any) -> Any:
        return input_ + tree_flatten(output_)[0][0]
    
    def input_hook_fn(model_: nn.Module, input_: Any, output_: Any) -> Any:
        return tree_flatten(input_)[0][0].cpu()
    
    model_type = "clip"
    if model_type == "clip":
        base_model_name = "openai/clip-vit-large-patch14"
        
        from transformers import CLIPVisionModel, CLIPImageProcessor
        model = CLIPVisionModel.from_pretrained(base_model_name)
        image_processor = CLIPImageProcessor.from_pretrained(base_model_name)
        transform = lambda images: image_processor(images=images, return_tensors="pt")["pixel_values"].squeeze(0)
        
        def fc_no_bias_hook_fn(model_: nn.Module, input_: Any, output_: Any) -> Any:
            return (tree_flatten(input_)[0][0] @ model_.weight.mT).cpu()
        
        monitor_config = OrderedDict({
            "vision_model.encoder.layers": OrderedDict({
                "": "layer_output",
                # "layer_norm1": "layer_norm1",  # "norm1"
                # "self_attn": "attention",   # "attention"
                # "layer_norm2": [
                #     ("layer_norm2_input", input_hook_fn),
                #     ("layer_norm2_output", Monitor.default_hook_fn),  # "norm2"
                # ],
                "mlp": {
                    "fc1": [
                        ("mlp_fc1_input", input_hook_fn),
                        # ("mlp_fc1_output_no_bias", fc_no_bias_hook_fn),
                        ("mlp_fc1_output", Monitor.default_hook_fn),
                    ],
                    # "activation_fn": [
                    #     ("mlp_activation", Monitor.default_hook_fn),
                    # ],
                    # "fc2": "mlp_fc2",
                }
            })
        })

    model = model.to(DEVICE)
    
    # SECTION: Dataset setup
    dataset = ImageDataset(dataset_name, transform, split="train", return_original_image=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True, generator=torch.Generator(DEVICE))
    
    # SECTION: Experiment setup
    monitor = Monitor(model, monitor_config)
    
    # output_dict = monitor.reset(return_mode="array")
    # with torch.no_grad():
    #     output = model("a photo of an astronaut riding a horse on mars")
    
    # print(output_dict.keys())
    # downblock_attention_output = output_dict["downblock_attention"]
    # print(type(downblock_attention_output), downblock_attention_output.shape)
    # # print([block_output[0].shape for block_output in downblock_attention00_output])
        
    # raise Exception()
    
    original_images, images = next(iter(dataloader))
    visualize_model_outputs(monitor, original_images, images)
    raise Exception()





# %%
