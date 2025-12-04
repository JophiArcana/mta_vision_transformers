#%%
import os
import json
import time
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from infrastructure import utils
from infrastructure.settings import DEVICE
# from modeling.vit_nystrom import LlavaNextNystromCompressionViT


def run_vqa_evaluation(
    model, # : LlavaNextNystromCompressionViT,
    model_name: str,
    experiment_dir: str,
    dataSubType: str = "val2014",
) -> List[Dict[str, Any]]:
    print(f"Evaluating {model_name}")

    MAX_NEW_TOKENS = 1024

    # SECTION: Set up evaluation files
    os.makedirs(experiment_dir, exist_ok=True)
    
    def f(s: str) -> str:
        return os.path.join(os.path.dirname(__file__), s)
    questions = json.load(open(f("v2_OpenEnded_mscoco_val2014_questions.json"), "r"))["questions"]
    
    output_fname = f"{experiment_dir}/{model_name}_output.jsonl"
    vqa_eval_fname = f"{experiment_dir}/{model_name}_vqa_obj.json"
    
    outputs = [json.loads(l) for l in open(output_fname, "r")] if os.path.exists(output_fname) else []
    vqa_eval_objs = json.load(open(vqa_eval_fname, "r")) if os.path.exists(vqa_eval_fname) else []

    output_fp = open(output_fname, "a") 
    for idx, question_idx in enumerate(tqdm([*range(0, len(questions), 100)])):
        question = questions[question_idx]
        if idx >= len(outputs):
            image_id = question["image_id"]
            image = Image.open(f"dataset/val2014/COCO_{dataSubType}_{str(image_id).zfill(12)}.jpg").convert("RGB")
            
            prompt = model.processor.apply_chat_template([{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question["question"]},
            ],},], add_generation_prompt=True)
            inputs = model.processor(image, prompt, return_tensors="pt").to(DEVICE)
            num_starting_tokens = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode(True):
                start_t = time.perf_counter()
                output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
                end_t = time.perf_counter()
            result: str = model.processor.decode(output[0], skip_special_tokens=True)
            answer = result.split("ASSISTANT:")[-1].strip()

            output = {
                "question_id": question["question_id"],
                "text": answer,
                "time": end_t - start_t,
                "num_starting_tokens": num_starting_tokens,
                "num_generated_tokens": output.shape[-1] - num_starting_tokens,
            }
            output_fp.write(json.dumps(output) + "\n")
            output_fp.flush()
            outputs.append(output)
        else:
            output = outputs[idx]
        
        if idx >= len(vqa_eval_objs):
            vqa_eval_objs.append({
                "question_id": output["question_id"],
                "answer": output["text"],
            })
            
        try:        
            model._cache.clear()
        except AttributeError:
            pass
        utils.empty_cache()

    vqa_eval_fp = open(vqa_eval_fname, "w")
    json.dump(vqa_eval_objs, vqa_eval_fp)
    vqa_eval_fp.close()

    output_fp.close()
    return outputs



