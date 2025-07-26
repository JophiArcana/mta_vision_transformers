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
        return tree_flatten(input_)[0][0]
    
    model_type = "clip"
    if model_type == "dino":
        base_model_name = "facebook/dino-vitb16"
        
        from transformers import ViTModel, ViTImageProcessor
        model = ViTModel.from_pretrained(base_model_name)
        image_processor = ViTImageProcessor.from_pretrained(base_model_name)
        transform = lambda images: image_processor(images=images, return_tensors="pt")["pixel_values"].squeeze(0)
    
    elif model_type == "dinov2":
        base_model_name = "facebook/dinov2-base"
        
        from transformers import Dinov2Model, AutoImageProcessor
        model = Dinov2Model.from_pretrained(base_model_name)
        image_processor = AutoImageProcessor.from_pretrained(base_model_name)
        transform = lambda images: image_processor(images=images, return_tensors="pt")["pixel_values"].squeeze(0)
        
        monitor_config = OrderedDict({
            "encoder.layer": OrderedDict({
                "": "layer_output",
                "layernorm_before": "layer_norm1",  # "norm1"
                "attention": "attention",   # "attention"
                "layernorm_after": "layer_norm2",  # "norm2"
                "mlp": {
                    "fc1": [
                        ("mlp_fc1_output", Monitor.default_hook_fn),
                    ],
                    "activation": [
                        ("mlp_activation_output", Monitor.default_hook_fn),
                    ],
                    "fc2": "mlp_fc2",
                }
            })
        })
        
    elif model_type == "clip":
        base_model_name = "openai/clip-vit-large-patch14"
        
        from transformers import CLIPVisionModel, CLIPImageProcessor
        model = CLIPVisionModel.from_pretrained(base_model_name)
        image_processor = CLIPImageProcessor.from_pretrained(base_model_name)
        transform = lambda images: image_processor(images=images, return_tensors="pt")["pixel_values"].squeeze(0)
        
        monitor_config = OrderedDict({
            "vision_model.encoder.layers": OrderedDict({
                "": "layer_output",
                "layer_norm1": "layer_norm1",  # "norm1"
                "self_attn": "attention",   # "attention"
                "layer_norm2": [
                    ("layer_norm2_output", Monitor.default_hook_fn),  # "norm2"
                    ("layer_norm2_input", input_hook_fn)
                ],
                "mlp": {
                    "fc1": [
                        ("mlp_fc1", Monitor.default_hook_fn),
                    ],
                    "activation_fn": [
                        ("mlp_activation", Monitor.default_hook_fn),
                    ],
                    "fc2": "mlp_fc2",
                }
            })
        })
        
    elif model_type == "open_clip":
        base_model_name = "laion2b_s32b_b82k"
        
        import open_clip
        model, _, transform = open_clip.create_model_and_transforms("ViT-L-14", pretrained=base_model_name)
        
        monitor_config = OrderedDict({
            "visual.transformer.resblocks": OrderedDict({
                "": "layer_output",
                "ln_1": "layer_norm1",  # "norm1"
                "attn": "attention",   # "attention"
                "ln_2": "layer_norm2",  # "norm2"
                "mlp": "mlp",
            })
        })
        
    elif model_type == "vitmae":
        base_model_name = "facebook/dinov2-base"
        
        from transformers import ViTMAEModel, VitMatteImageProcessor
        model = ViTMAEModel.from_pretrained(base_model_name)
        image_processor = VitMatteImageProcessor.from_pretrained(base_model_name)
        transform = lambda images: image_processor(images=images, return_tensors="pt")["pixel_values"].squeeze(0)
        
    elif model_type == "sam":
        base_model_name = "facebook/sam-vit-base"
        
        from transformers import SamModel, SamImageProcessor
        model = SamModel.from_pretrained(base_model_name)
        image_processor = SamImageProcessor.from_pretrained(base_model_name)
        transform = lambda images: image_processor(images=images, return_tensors="pt")["pixel_values"].squeeze(0)
        
    elif model_type == "dit":
        # base_model_name = "stabilityai/stable-diffusion-3.5-medium"
        base_model_name = "sd-legacy/stable-diffusion-v1-5"
        
        from diffusers import StableDiffusionPipeline
        model = StableDiffusionPipeline.from_pretrained(base_model_name)
        transform = None
        
        monitor_config = OrderedDict({
            # "unet.down_blocks": "downblock",
            # "unet.up_blocks": "upblock",
            "unet.down_blocks.attentions": "downblock_attention",
            "unet.up_blocks.attentions": "upblock_attention",
        })
        
        # model = BeitModel.from_pretrained(base_model_name)
        # image_processor = BeitImageProcessor.from_pretrained(base_model_name)
        # transform = lambda images: image_processor(images=images, return_tensors="pt")["pixel_values"].squeeze(0)
        
    model = model.to(DEVICE)
    
    # SECTION: Dataset setup
    dataset = ImageDataset(dataset_name, transform, split="train", return_original_image=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True, generator=torch.Generator(DEVICE))
    
    # SECTION: Experiment setup
    monitor = Monitor(model, monitor_config)
    # monitor = Monitor(model, OrderedDict({
    #     "visual.transformer.resblocks": OrderedDict({
    #         "": "layer_output",
    #         "ln_1": "layer_norm1",  # "norm1"
    #         "attn": "attention",   # "attention"
    #         "ln_2": "layer_norm2",  # "norm2"
    #         "mlp": "mlp",
    #     })
    # }))
    
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

    def process_grayscale(im):
        arr = np.array(im)
        return arr if arr.ndim >= 3 else np.tile(arr[..., None], (1, 1, 3))
    images = [*map(process_grayscale, images)]

    # SECTION: Debugging
    from transformers import ViTModel, ViTImageProcessor
    from model.multistate_encoder.modeling_msvitencoder import MultiStateViTConfig, MultiStateViTEncoderModel
    from model.clustering.modeling_fps import FPSClusteringConfig
    
    torch.manual_seed(1212)
    torch.cuda.empty_cache()
    base = ViTModel.from_pretrained(base_model_name)

    image_size = 224
    image_processor = ViTImageProcessor.from_pretrained(base_model_name)
    image_processor.__dict__.update({
        "size": {"height": image_size, "width": image_size},
    })
    inputs = image_processor(images=images, return_tensors="pt")

    model = MultiStateViTEncoderModel(MultiStateViTConfig(
        **base.config.to_dict(),
        _attn_implementation="eager",
        pregeneration_period=10,
        generation_period=2,
        clustering_config=FPSClusteringConfig(
            ncut_dim=100,
            fps_dim=8,
            fps_sample1=300,
            fps_sample2=100,
            fps_supersample2=120,
            cosine_similarity_threshold=0.7,
        ),
        pretrained=base_model_name
    ))
    print(model)
    print(model.config)
    # for image in images[:3]:
    #     plt.imshow(image)
    #     plt.show()
    with torch.no_grad():
        print(model(**inputs, interpolate_pos_encoding=True))
    raise Exception()

    # SECTION: Model setup
    model = CLIPVisionModel.from_pretrained(model_name)
    # print(model)

    print(model)
    print(model.config)

    # model.embeddings(**inputs)
    a = model.get_input_embeddings()(inputs["pixel_values"])
    print(a.dtype, a.shape)

    print()
    print(model.vision_model.embeddings(inputs["pixel_values"]).shape)
    raise Exception()

    affinity_focal_gamma = 1.0
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        for layer, states in enumerate(hidden_states):
            X = states[..., 1:, :].flatten(0, -2)

            normalized_X = torch.nn.functional.normalize(X, dim=-1)
            normalized_A = 1.0 - normalized_X @ normalized_X.mT
            A = (X.norm(dim=-1)[:, None] * X.norm(dim=-1)[None, :]) * normalized_A

            A = torch.exp(-A / affinity_focal_gamma)

            D = A.sum(dim=-1)
            L = torch.eye(len(D)) - A * ((D[:, None] * D[None, :]) ** -0.5)

            E, V = torch.linalg.eigh(L)
            X = V[:, :10]

            X_embedded = TSNE(n_components=2).fit_transform(X)

            plt.scatter(*X_embedded.T)
            plt.title(f"Layer {layer}")
            plt.show()

            print(layer, states.shape, X_embedded.shape)






# %%
