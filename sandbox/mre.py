import sys
sys.path.append("/workspace/mta_vision_transformers/")

import torch

from modeling.base_vit import OpenCLIPViT, DINOv2ViT
from modeling.vit_nystrom import OpenCLIPNystromCompressionViT, DINOv2NystromCompressionViT


if __name__ == "__main__":
    torch.set_printoptions(linewidth=400, sci_mode=False)

    modes = ["fps", "uniform", "multiclass_spectral_clustering", "kmeans", "segment_means", "spectral_clustering"]
    mode = modes[0]
    # default kwargs:
    #     compression_mode = "nystrom"
    #     num_sample: int = 32
    #     resample: bool = True
    #     use_layer_input: bool = True
    #     mask_layers = range(13, 24) # range(13, ImageFeatures.NUM_LAYERS + 1)

    baseline_clip = OpenCLIPViT()
    baseline_dinov2 = DINOv2ViT()
    nystrom_clip = OpenCLIPNystromCompressionViT(mode)
    nystrom_dinov2 = DINOv2NystromCompressionViT(mode)
    
    model_list = [baseline_clip, baseline_dinov2, nystrom_clip, nystrom_dinov2]
    pixel_values = torch.randn((32, 3, 224, 224))
    
    for model in model_list:
        print(model)
        print(model(pixel_values))
        print("\n" * 10)
    




