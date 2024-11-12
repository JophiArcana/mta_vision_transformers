from typing import Any, Callable, Dict, List, Set

import einops
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ncut_pytorch import NCUT, rgb_from_tsne_3d
from torch.utils._pytree import tree_flatten

from infrastructure.settings import *
from core.monitor import Monitor


def visualize_model_outputs(
    monitor: Monitor,
    original_images: torch.Tensor,
    input_images: torch.Tensor,
    num_visualized_images: int = 8,
) -> None:
    bsz = input_images.shape[0]
    model = monitor.model
    
    def shift_channels(images_: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(images_, "bsz c h w -> bsz h w c")
    
    # SECTION: Visualize original images
    fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(4 * num_visualized_images, 4))
    for image_idx, original_image in enumerate(shift_channels(original_images[:num_visualized_images])):
        axs[image_idx].imshow(original_image.numpy(force=True))
        axs[image_idx].axis("off")
    fig.suptitle("original_image")
    plt.show()
    
    output_dict = monitor.reset()
    with torch.no_grad():
        output = model.forward(input_images)
    
    layer_output_dicts = [
        dict(zip(output_dict.keys(), layer_outputs))
        for layer_outputs in zip(*output_dict.values())
    ]
    
    H = W = 16
    ncut = NCUT(num_eig=100, distance="rbf", indirect_connection=False, device=DEVICE)
    def visualize_features(metric_name: str, t: torch.Tensor) -> None:
        ncut_features, eigenvalues = ncut.fit_transform(t.flatten(0, 1))
        rgb_features = rgb_from_tsne_3d(ncut_features)[1]
        rgb_features = einops.rearrange(
            rgb_features.reshape(bsz, -1, 3)[:, 1:],
            "bsz (h w) c -> bsz h w c", h=H, w=W
        )
        
        fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(4 * num_visualized_images, 4))
        for image_idx, image_features in enumerate(rgb_features[:num_visualized_images]):
            axs[image_idx].imshow(image_features.numpy(force=True))
            axs[image_idx].axis("off")
        fig.suptitle(metric_name)
        plt.show()
    
    def visualize_feature_norms(metric_name: str, t: torch.Tensor) -> None:
        feature_norms = einops.rearrange(
            torch.norm(t, dim=-1, p=2)[:, 1:],
            "bsz (h w) -> bsz h w", h=H, w=W
        )
        assert torch.all(feature_norms >= 0), "Computed norms should be greater than 0."
        
        fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(4 * num_visualized_images, 4))
        for image_idx, image_norms in enumerate(feature_norms[:num_visualized_images]):
            axs[image_idx].imshow(image_norms.numpy(force=True), cmap="gray")
            axs[image_idx].axis("off")
        fig.suptitle(f"{metric_name}_norm")
        plt.show()
    
    
    include: Set[str] = {"layer_output"}
    for layer_idx, layer_output_dict in enumerate(layer_output_dicts):
        print(f"Layer {layer_idx} {'=' * 120}")
        
        for metric, layer_output in layer_output_dict.items():
            metric_outputs, _ = tree_flatten(layer_output)
            if metric in include:
                visualize_features(metric, metric_outputs[0])
            visualize_feature_norms(metric, metric_outputs[0])

    raise Exception()
    



