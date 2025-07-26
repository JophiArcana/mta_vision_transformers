#%%
import os
import json
import sys
from tqdm import tqdm
from typing import Any, Dict, List, Set, Tuple
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir("/workspace/mta_vision_transformers/")
sys.path.append("/workspace/mta_vision_transformers/")

from matplotlib import pyplot as plt
from PIL import Image, ImageFile


def run_vqa_evaluation(
    model_names: List[str],
    indices: Set[int] = {*range(3, 6)},
) -> List[Tuple[ImageFile.ImageFile, str, str, List[str]]]:

    # SECTION: Set up evaluation files
    rootdir = "playground/data/coco2014_val_qa_eval"
    def f(s: str) -> str:
        return f"{rootdir}/qa90_{s}.jsonl"
    os.makedirs(rootdir, exist_ok=True)
    
    print(f("gpt4_answer"))
    print(os.path.exists(f("gpt4_answer")))
    print(os.getcwd())
    target_fp = open(f("gpt4_answer"), "r")
    question_fp = open(f("questions"), "r")

    result = []
    for idx, pair in tqdm(enumerate(zip(target_fp, question_fp))):
        target, question = json.loads(pair[0]), json.loads(pair[1])

        # SECTION: Run the model to obtain the answer to be evaluated
        if idx in indices:
            image = Image.open(f"dataset/val2014/COCO_val2014_{question['image']}").convert("RGB")
            outputs = []
            for model_name in model_names:
                outputs.append([json.loads(l) for l in open(f(f"{model_name}_answer"), "r")][idx]["text"])
                
            result.append((image, question["text"], target["text"], outputs))
    
    return result


if __name__ == "__main__":
    # SECTION: Run model on VQA
    
    model_names = ["baseline", "L[18:32]_S256"]
    
    indices = [*range(0, 90, 3)]
    result = run_vqa_evaluation(model_names, indices)
    
    for idx, (im, question, target, outputs) in zip(indices, result):
        plt.axis("off")
        plt.imshow(im)
        plt.show()

        im.save(f"im{idx}.png")
        
        print(f"Idx: {idx}")
        print(f"Question: {question}")
        print(f"Target: {target}")
        for m, o in zip(model_names, outputs):
            print(f"{m}: {o}")
        print("\n" * 5)
    


# %%
