#%%
import inspect
import itertools
import os
import requests
import sys
import time
from collections import OrderedDict
from typing import List, Tuple
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("/workspace/mta_vision_transformers/")

import einops
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image, ImageFile

from core.monitor import Monitor
from infrastructure import utils
from infrastructure.settings import DEVICE, OUTPUT_DEVICE
from modeling.base_vit import OpenCLIPViT, DINOv2ViT
from modeling.vit_nystrom import OpenCLIPNystromCompressionViT, DINOv2NystromCompressionViT


def factor_image_shape(t: int, s: Tuple[int, int]) -> Tuple[int, int]:
    h, w = s
    estimate_h = round((t * h / w) ** 0.5)
    for _h in range(estimate_h - 2, estimate_h + 3):
        if t % _h == 0:
            return _h, (t // _h)
    raise Exception("Factorization not found")

def plot_image_row(images, permute=False, unnormalize=False, title=None, save_path=None, cmap=None):
    # Display the first 8 images in a row
    fig, axes = plt.subplots(1, 8, figsize=(14, 4))
    for i in range(8):
        image = images[i]
        if permute:
            image = image.permute(1, 2, 0)
        if unnormalize:
            image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            image = image.clamp(0, 1)
        axes[i].imshow(image, cmap=cmap)
        axes[i].axis('off')
        
    plt.tight_layout()  # Tight layout without considering suptitle
    if title:
        fig.suptitle(title, fontsize=16, y=0.8)  # y controls vertical position of the title
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close()


if __name__ == "__main__":
    torch.set_printoptions(linewidth=400, sci_mode=False)
    

    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    from transformers.tokenization_utils_base import BatchEncoding

    # Load the model in half-precision
    name = "llava-hf/llava-v1.6-vicuna-7b-hf"
    model = LlavaNextForConditionalGeneration.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto")
    processor = LlavaNextProcessor.from_pretrained(name)

    monitor_config = OrderedDict({
        "language_model.model.layers": OrderedDict({
            "": [
                ("layer_output", Monitor.default_hook_fn),
            ],
            "self_attn": [
                ("attention_output", Monitor.default_hook_fn),
            ],
        })
    })
    monitor = Monitor(model, monitor_config)

    images = [
        Image.open(f"sandbox/images/image_{i}.png")
        for i in range(16)
    ]
    prompt = processor.apply_chat_template([{"role": "user", "content": [
        {"type": "image",},
        {"type": "text", "text": "",},
    ],},], add_generation_prompt=True)


    layer_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    attention_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for image in images:
        
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
        mask = (inputs["input_ids"][0] == model.config.image_token_index)
        
        log = monitor.reset()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1,)

        def process_image_features(key: str) -> Tuple[torch.Tensor, torch.Tensor]:
            features = torch.stack([l[0][0] for l in log[key]], dim=1)
            image_feature = features[mask.to(OUTPUT_DEVICE)]
            
            lo = einops.rearrange(
                image_feature[:576], "(h w) l d -> l h w d",
                h=24, w=24,
            )
            W, H = image.size
            h, w = factor_image_shape(image_feature.shape[0] - 576, (H, W))
            hi = einops.rearrange(
                image_feature[576:], "(h w) l d -> l h w d",
                h=h, w=w,
            )[:, :h, :-1, :]
            return lo, hi
        
        layer_outputs.append(process_image_features("layer_output"))
        attention_outputs.append(process_image_features("attention_output"))

    layer_outputs: Tuple[List[torch.Tensor], List[torch.Tensor]] = [*zip(*layer_outputs)]
    attention_outputs: Tuple[List[torch.Tensor], List[torch.Tensor]] = [*zip(*attention_outputs)]

    layer_outputs[0] = torch.stack(layer_outputs[0], dim=0)
    attention_outputs[0] = torch.stack(attention_outputs[0], dim=0)
    
    
    # from nystrom_ncut import NystromNCut, KernelNCut, rgb_from_euclidean_tsne_3d, SampleConfig
    from ncut_pytorch import NCUT, rgb_from_tsne_3d

    for layer_idx in [16]:
        # nc = NystromNCut(n_components=100, affinity_type="rbf", sample_config=SampleConfig())
        nc = NCUT(n_components=100, distance="rbf", num_sample=20000)
        
        def do_visualization(seq: Tuple[torch.Tensor, List[torch.Tensor]], name: str):
            lo = seq[0][:, layer_idx, :, :, :]              # float: [bsz x h x w x D]
            hi = [t[layer_idx, :, :, :] for t in seq[1]]    # float: bsz x [H x W x D]
            tensors = [lo] + hi
            shapes = [t.shape[:-1] for t in tensors]
            cs = [0] + np.cumsum([np.prod(s) for s in shapes]).tolist()
            
            all_tokens = torch.cat([t.flatten(0, -2) for t in [lo] + hi], dim=0)
            features = nc.fit_transform(all_tokens.to(device=DEVICE, dtype=torch.float32))[0]
            rgb = rgb_from_tsne_3d(features, num_sample=3000)[0]

            colors = [rgb[l:r].reshape((*shape, -1)) for shape, (l, r) in zip(shapes, itertools.pairwise(cs))]
            lo_rgb = colors[0]
            hi_rgb = colors[1:]
            
            plot_image_row([*lo_rgb], title=f"Layer {layer_idx} {name} Low Resolution - NCut Features")
            plot_image_row(hi_rgb, title=f"Layer {layer_idx} {name} High Resolution - NCut Features")

        do_visualization(layer_outputs, "Transformer Output")
        raise Exception()
        



# %%
