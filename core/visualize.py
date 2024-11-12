import itertools
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Set, Tuple

import einops
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ncut_pytorch import NCUT, rgb_from_tsne_3d
from torch.utils._pytree import tree_flatten

from infrastructure.settings import DEVICE
from infrastructure import utils
from core.monitor import Monitor


def visualize_model_outputs(
    monitor: Monitor,
    original_images: torch.Tensor,
    input_images: torch.Tensor,
    num_visualized_images: int = 8,
) -> None:
    bsz = input_images.shape[0]
    H = W = 16
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
        
    stacked_layer_output_dict = OrderedDict([
        (metric, torch.stack([
            metric_output[0]
            for metric_output in metric_outputs
        ], dim=0))
        for metric, metric_outputs in output_dict.items()
    ])
    
    layer_output_dicts = [
        OrderedDict(zip(output_dict.keys(), layer_outputs))
        for layer_outputs in zip(*output_dict.values())
    ]
    
    
    # SECTION: Massive token heuristic
    def massive_token_heuristic(stacked_layer_output_dict: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        log_norms = einops.rearrange(
            torch.norm(stacked_layer_output_dict["layer_output"], p=2, dim=-1).log()[14, :, 1:],
            "bsz (h w) -> bsz h w", h=H, w=W
        )
        flattened_norms = torch.sort(torch.flatten(log_norms), dim=0).values
        cutoff = torch.argmax(torch.diff(flattened_norms, dim=0), dim=0)
        mask = log_norms > flattened_norms[cutoff]
        return mask
    mta_mask = massive_token_heuristic(stacked_layer_output_dict)
    
    
    # SECTION: Per image visualization code
    ncut = NCUT(num_eig=100, distance="rbf", indirect_connection=False, device=DEVICE)
    def visualize_features_per_image(metric_name: str, t: torch.Tensor, plot: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        ncut_features, eigenvalues = ncut.fit_transform(t.flatten(0, 1))
        rgb_features = rgb_from_tsne_3d(ncut_features)[1]
        rgb_features = einops.rearrange(
            rgb_features.reshape(bsz, -1, 3)[:, 1:],
            "bsz (h w) c -> bsz h w c", h=H, w=W
        )
        
        if plot:
            fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(4 * num_visualized_images, 4))
            for image_idx, image_features in enumerate(rgb_features[:num_visualized_images]):
                axs[image_idx].imshow(image_features.numpy(force=True))
                axs[image_idx].axis("off")
            fig.suptitle(metric_name)
            plt.show()

        return einops.rearrange(
            ncut_features.unflatten(0, (bsz, -1))[:, 1:],
            "bsz (h w) c -> bsz h w c", h=H, w=W
        ), rgb_features
    
    def visualize_feature_norms_per_image(metric_name: str, t: torch.Tensor) -> torch.Tensor:
        feature_norms = einops.rearrange(
            torch.norm(t, p=2, dim=-1)[:, 1:],
            "bsz (h w) -> bsz h w", h=H, w=W
        )
        assert torch.all(feature_norms >= 0), "Computed norms should be greater than 0."
        
        fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(4 * num_visualized_images, 4))
        for image_idx, image_norms in enumerate(feature_norms[:num_visualized_images]):
            axs[image_idx].imshow(image_norms.numpy(force=True), cmap="gray")
            axs[image_idx].axis("off")
        fig.suptitle(f"{metric_name}_norm")
        plt.show()
        return feature_norms
    
    
    # SECTION: Per layer visualization code
    _, rgb_assignment = visualize_features_per_image(None, layer_output_dicts[-1]["layer_output"][0], plot=False)
    

    def visualize_feature_norms_per_layer(metric_name: str, t: torch.Tensor) -> torch.Tensor:
        feature_norms = torch.norm(t, p=2, dim=-1)
        for image_idx, token_idx in itertools.product(range(bsz), range(-1, H * W)):
            h_idx, w_idx = token_idx // W, token_idx % W
            token_feature_norms = feature_norms[:, image_idx, token_idx + 1]
            
            if token_idx == -1:
                plt.plot(
                    token_feature_norms.numpy(force=True), marker=".",
                    color="black", zorder=12, label="cls_token" if image_idx == 0 else None
                )
            else:
                if mta_mask[image_idx, h_idx, w_idx]:
                    mta_kwargs = {"linewidth": 1, "linestyle": "-",}
                else:
                    mta_kwargs = {"linewidth": 0.5, "linestyle": "-.",}
                    
                plt.plot(
                    token_feature_norms.numpy(force=True), marker=".",
                    color=rgb_assignment[image_idx, h_idx, w_idx].numpy(force=True), **mta_kwargs,
                )
        
        plt.title(f"{metric_name}_norm")
        plt.xlabel("layer")
        plt.gca().xaxis.grid(True)
        plt.ylabel("norm")
        plt.yscale("log")
        
        plt.legend()
        plt.show()
        
    for metric, stacked_metric_output in stacked_layer_output_dict.items():
        visualize_feature_norms_per_layer(metric, stacked_metric_output)
    raise Exception()
    
    stacked_layer_outputs = einops.rearrange(
        torch.stack([
            layer_output_dict["layer_output"][0]
            for layer_output_dict in layer_output_dicts
        ], dim=0)[:, :, 1:],
        "l bsz (h w) c -> l bsz h w c", h=H, w=W
    )
    
    # SECTION: Visualize layer output norms across layers
    for token, rgb in zip(
        torch.unbind(stacked_layer_outputs.flatten(-4, -2), dim=-2),
        torch.unbind(rgb_assignment.flatten(-4, -2), dim=-2),
    ):
        plt.plot(torch.norm(token, dim=-1).numpy(force=True), color=rgb.numpy(force=True))
    plt.xlabel("layer")
    plt.ylabel("layer_output_norm")
    plt.yscale("log")
    plt.show()
        
    print(rgb_assignment.shape, stacked_layer_outputs.shape)


    include: Set[str] = {"layer_output"}
    rgb_assignment: torch.Tensor = None
    for layer_idx, layer_output_dict in enumerate(layer_output_dicts):
        print(f"Layer {layer_idx} {'=' * 120}")
        
        feature_norm_dict: Dict[str, torch.Tensor] = {}
        for metric, metric_output in layer_output_dict.items():
            if metric in include:
                visualize_features_per_image(metric, metric_output[0])
            feature_norm_dict[metric] = visualize_feature_norms_per_image(metric, metric_output[0])
        
        x_metric, y_metric = "layer_norm1", "attention"
        plt.scatter(
            feature_norm_dict[x_metric].numpy(force=True),
            feature_norm_dict[y_metric].numpy(force=True),
            color="black", s=2
        )
        plt.title(f"Layer {layer_idx}")
        plt.xlabel(f"{x_metric}_norm")
        plt.ylabel(f"{y_metric}_norm")
        plt.show()
        
        # if layer_idx == len(layer_output_dicts) - 1:
            
        


    raise Exception()
    



