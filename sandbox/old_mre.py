import inspect
import os
import requests
import sys
import time
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("/workspace/mta_vision_transformers/")

import torch
import torch.nn as nn
from PIL import Image

from infrastructure import utils
from infrastructure.settings import DEVICE
from modeling.base_vit import OpenCLIPViT, DINOv2ViT
from modeling.vit_nystrom import OpenCLIPNystromCompressionViT, DINOv2NystromCompressionViT


if __name__ == "__main__":
    torch.set_printoptions(linewidth=400, sci_mode=False)
    

    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    from transformers.tokenization_utils_base import BatchEncoding

    # Load the model in half-precision
    # model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=DTYPE, device_map="auto")
    # processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    from modeling.vit_nystrom import LlavaNextNystromCompressionViT

    model = LlavaNextNystromCompressionViT("fps", mask_layers=range(16, 32), num_sample=128)
    processor = model.processor

    # prepare image and text prompt, using the appropriate prompt template
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(image, prompt, return_tensors="pt").to(DEVICE)
    num_starting_tokens = inputs["input_ids"].shape[-1]

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=500)

    print(processor.decode(output[0], skip_special_tokens=True))
    raise Exception()




    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]

    inputs: BatchEncoding = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    # Generate
    generate_ids = model.generate(**inputs, max_new_tokens=30)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True)
    
    print("\n" * 10)
    print(output)
    
    raise Exception()
    
    
    
    
    
    
    
    

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
    
    
    
    
    
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What’s shown in this image?"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "This image shows a red stop sign."},
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in more details."},
            ],
        },
    ]




