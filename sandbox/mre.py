import sys
sys.path.append("/workspace/mta_vision_transformers/")

import torch

from infrastructure.settings import DEVICE
from modeling.base_vit import OpenCLIPViT, DINOv2ViT
from modeling.vit_nystrom import OpenCLIPNystromCompressionViT, DINOv2NystromCompressionViT


if __name__ == "__main__":
    torch.set_printoptions(linewidth=400, sci_mode=False)
    
    
    
    
    import torch.nn as nn
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    model_id = "stabilityai/stable-diffusion-2-1"

    # Use the Euler scheduler here instead
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    
    import inspect
    from transformers.models.clip.modeling_clip import CLIPTextTransformer
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    from diffusers.models.attention import BasicTransformerBlock
    # print(CLIPTextTransformer, pipe.text_encoder)
    # print("\n" * 12)
    # print(pipe.unet)
    print(pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0])
    print(pipe.unet.mid_block.attentions[0].transformer_blocks[0])
    print(pipe.unet.up_blocks[-1].attentions[0].transformer_blocks[0])
    print(inspect.getfile(type(pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0])))
    # unet: nn.Module = pipe.unet
    # for k, v in unet.named_parameters():
    #     if "attention" in k:
    #         print(k)
    raise Exception()

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]
        
    image.save("astronaut_rides_horse.png")
    raise Exception()

    # image = pipe(
    #     "A capybara holding a sign that reads Hello World",
    #     num_inference_steps=28,
    #     guidance_scale=3.5,
    # ).images[0]
    # image.save("capybara.png")
    
    
    
    
    # from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor

    # # Load the model in half-precision
    # model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
    # processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    # print(processor)
    # raise Exception()

    # conversation = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
    #             {"type": "text", "text": "What is shown in this image?"},
    #         ],
    #     },
    # ]

    # inputs = processor.apply_chat_template(
    #     conversation,
    #     add_generation_prompt=True,
    #     tokenize=True,
    #     return_dict=True,
    #     return_tensors="pt"
    # ).to(model.device, torch.float16)

    # # Generate
    # generate_ids = model.generate(**inputs, max_new_tokens=30)
    # processor.batch_decode(generate_ids, skip_special_tokens=True)
    
    
    
    
    
    
    
    

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




