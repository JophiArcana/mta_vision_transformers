#%%
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("/workspace/mta_vision_transformers/")

import torch
from matplotlib import pyplot as plt

from evaluation.coco_vqa.lib import run_vqa_evaluation
from modeling.vit_nystrom import LlavaNextNystromCompressionViT


if __name__ == "__main__":
    # SECTION: Run model on VQA
    
    model = LlavaNextNystromCompressionViT("fps", mask_layers=range(32, 32), num_sample=256)
    run_vqa_evaluation(model, "baseline", "evaluation/coco_vqa/test")
    raise Exception()
    
    configs = [
        (18, 32, 256,),
        (18, 32, 128,),
        (18, 32, 64,),
        (18, 32, 32,),
    ]
    model_kwargs_dict = {
        "baseline": {"mask_layers": range(0, 0),},
        **{
            f"layers[{lo}:{hi}]_sample{num_sample}": {"mask_layers": range(lo, hi), "num_sample": num_sample,}
            for lo, hi, num_sample in configs
        },
    }
    
    for model_name, model_kwargs in model_kwargs_dict.items():
        model = None # LlavaNextNystromCompressionViT("fps", **model_kwargs)
        outputs, reviews = run_vqa_evaluation(model, model_name, None)
        
        ldict = dict(zip(outputs[0].keys(), zip(*map(dict.values, outputs))))
        y, x = torch.tensor(ldict["time"], dtype=torch.float32), torch.tensor(ldict["num_generated_tokens"], dtype=torch.float32)

        X = torch.stack((x, torch.ones_like(x)), dim=-1)
        lstsq = torch.linalg.pinv(X) @ y
        xl = torch.linspace(-0.1, 1.1, 100) * (x.max() - x.min()) + x.min()
        yl = lstsq[0] * xl + lstsq[1]

        plt.scatter(
            x.numpy(force=True), y.numpy(force=True),
            s=10, label=f"{model_name}: $y = {lstsq[0].item():.4f}x + {lstsq[1].item():.4f}$",
        )
        plt.plot(
            xl.numpy(force=True), yl.numpy(force=True),
            color="black", linewidth=1.0, linestyle="--",
        )
        
        del model
        utils.empty_cache()
    
    plt.xlabel("num_generated_tokens")
    plt.ylabel("time (s)")
    plt.legend()
    
    plt.show()






    # hi = 32
    # for lo in range(0, hi, 2):
    #     for num_sample in [1 << k for k in range(4, 11)]:
    #         model = LlavaNextNystromCompressionViT("fps", mask_layers=range(lo, hi))
    #         output_fname, review_fname = run_vqa_evaluation(model, f"layers[{lo}:{hi}]_sample{num_sample}")
    #         del model
    #         utils.empty_cache()

# %%
