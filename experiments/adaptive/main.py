#%%
import sys
sys.path.append("/workspace/mta_vision_transformers/")

import torch
import torch.utils.data

from infrastructure.settings import DEVICE
from dataset.evaluation import run_retrieval_evaluation
from modeling.openclip_vit import OpenCLIPViT
from modeling.vit_adaptive import OpenCLIPAdaptiveViT


if __name__ == "__main__":
    torch.set_printoptions(linewidth=400, sci_mode=False)

    # Run evaluation
    # adaptive_model = OpenCLIPViT().to(DEVICE)
    adaptive_model = OpenCLIPAdaptiveViT("sink", mask_layer=13, reset_layer=9, detection_layer=13).to(DEVICE)

    # Evaluate adaptive model
    print("=" * 120)
    print("Adaptive model")
    print("=" * 120)
    adaptive_retrieval_metrics = run_retrieval_evaluation(adaptive_model)

    # Print results
    print("Text-to-Image Retrieval Metrics:")
    for k, value in adaptive_retrieval_metrics["text_to_image"].items():
        print(f"{k}: {value:.2f}%")

    print("\nImage-to-Text Retrieval Metrics:")
    for k, value in adaptive_retrieval_metrics["image_to_text"].items():
        print(f"{k}: {value:.2f}%")


# %%
