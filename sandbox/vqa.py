#%%
import os
import json
import sys
import time
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("/workspace/mta_vision_transformers/")

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
# from openai import OpenAI, RateLimitError
from PIL import Image

from infrastructure import utils
from infrastructure.settings import DEVICE
from modeling.vit_nystrom import LlavaNextNystromCompressionViT


def get_gpt4_review(content: str, max_tokens: int) -> str:
    client = OpenAI()
    NUM_SECONDS_TO_SLEEP = 0.5
    while True:
        try:
            response = client.chat.completions.create(model="gpt-4-0613",
            messages=[{
                "role": "system",
                "content": "You are a helpful and precise assistant for checking the quality of the answer."
            }, {
                "role": "user",
                "content": content,
            }],
            temperature=0.2,  # TODO: figure out which temperature is best for evaluation
            max_tokens=max_tokens)
            break
        except RateLimitError:
            pass
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response.choices[0].message.content

def parse_score(review: str) -> List[int]:
    score_pair = review.split("\n")[0]
    score_pair = score_pair.replace(",", " ").split(" ")
    assert len(score_pair) == 2, f"Number of scores should be 2 but got {2}"
    return [*map(float, score_pair)]

def run_vqa_evaluation(model: LlavaNextNystromCompressionViT, model_name: str) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, List[int]]
]:
    print(f"Evaluating {model_name}")

    # SECTION: Set up prompt contexts
    eval_rootdir = "llava/eval/table"
    rule_dict = json.load(open(f"{eval_rootdir}/rule.json", "r"))
    context_list = [json.loads(line) for line in open(f"{eval_rootdir}/caps_boxes_coco2014_val_80.jsonl")]
    image_to_context = {context["image"]: context for context in context_list}

    # SECTION: Set up evaluation files
    rootdir = "playground/data/coco2014_val_qa_eval"
    def f(s: str) -> str:
        return f"{rootdir}/qa90_{s}.jsonl"
    os.makedirs(rootdir, exist_ok=True)
    
    question_fp = open(f("questions"), "r")
    
    output_fname, review_fname, score_fname = f(f"{model_name}_answer"), f(f"gpt4_review_{model_name}"), f(f"BERTScore_{model_name}")
    target_fname = f("gpt4_answer")
     
    outputs = [json.loads(l) for l in open(output_fname, "r")] if os.path.exists(output_fname) else []
    # reviews = [json.loads(l) for l in open(review_fname, "r")] if os.path.exists(review_fname) else []
    score_dict: Dict[str, List[int]] = json.load(open(score_fname, "r")) if os.path.exists(score_fname) else {}
    
    n_questions = 90
    # if all(len(l) == n_questions for l in [outputs, reviews, *score_dict.values()]):
    if all(len(l) == n_questions for l in [outputs, *score_dict.values()]):
        # return outputs, reviews, score_dict
        return outputs, score_dict
    
    MAX_NEW_TOKENS = 1024
    processor = LlavaNextNystromCompressionViT.processor
    
    target_fp = open(target_fname, "r")
    output_fp = open(output_fname, "a")
    # review_fp = open(review_fname, "a")
    for idx, pair in tqdm(enumerate(zip(target_fp, question_fp))):
        target, question = json.loads(pair[0]), json.loads(pair[1])

        # SECTION: Run the model to obtain the answer to be evaluated
        if idx >= len(outputs):
            image = Image.open(f"dataset/val2014/COCO_val2014_{question['image']}").convert("RGB")
            prompt = processor.apply_chat_template([{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question["text"]},
            ],},], add_generation_prompt=True)
            inputs = processor(image, prompt, return_tensors="pt").to(DEVICE)
            num_starting_tokens = inputs["input_ids"].shape[-1]

            with torch.inference_mode(True):
                start_t = time.perf_counter()
                output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
                end_t = time.perf_counter()
            result: str = processor.decode(output[0], skip_special_tokens=True)

            output = {
                "question_id": question["question_id"],
                "text": result.split("ASSISTANT:")[-1].strip(),
                "time": end_t - start_t,
                "num_starting_tokens": num_starting_tokens,
                "num_generated_tokens": output.shape[-1] - num_starting_tokens,
            }
            output_fp.write(json.dumps(output) + "\n")
            output_fp.flush()
            outputs.append(output)
        else:
            output = outputs[idx]
        
        # SECTION: Get BERT score
        import evaluate
        
        keys = ("f1", "precision", "recall")
        if any(idx >= len(score_dict.get(k, [])) for k in keys):
            bertscore = evaluate.load("bertscore")
            scores = bertscore.compute(predictions=[output["text"]], references=[target["text"]], lang="en")
            for k in keys:
                score_dict.setdefault(k, []).extend(scores[k])
            
            with open(score_fname, "w") as score_fp:
                score_fp.write(json.dumps(score_dict))
            
            time.sleep(0.5)
        
        # # SECTION: Supply reviews via GPT-4 (0613)
        # if idx >= len(reviews):
        #     inst = image_to_context[question["image"]]
        #     cap_str = "\n".join(inst["captions"])
        #     box_str = "\n".join([f"{instance['category']}: {instance['bbox']}" for instance in inst['instances']])

        #     category = question["category"]
        #     rule = rule_dict[category]
        #     prompt = rule["prompt"]
        #     role = rule["role"]
            
        #     content = (f"[Context]\n{cap_str}\n\n{box_str}\n\n"
        #             f"[Question]\n{question['text']}\n\n"
        #             f"[{role}]\n{output['text']}\n\n[End of {role}]\n\n"
        #             f"[System]\n{prompt}\n\n")
        
        #     content = get_gpt4_review(content, MAX_NEW_TOKENS)
        #     score = float(content.split("\n")[0])
        #     review = {
        #         "question_id": question["question_id"],
        #         "category": category,
        #         "content": content,
        #         "score": score,
        #     }
        #     review_fp.write(json.dumps(review) + "\n")
        #     review_fp.flush()
        #     reviews.append(review)
        # else:
        #     review = reviews[idx]
        #     score = review["score"]
        
        # if idx >= len(score_dict.get("all", [])):
        #     score_dict.setdefault(review["category"], []).append(score)
        #     score_dict.setdefault("all", []).append(score)
        
        #     with open(score_fname, "w") as score_fp:
        #         score_fp.write(json.dumps(score_dict))

        try:        
            model._cache.clear()
        except AttributeError:
            pass
        utils.empty_cache()

    output_fp.close()
    # review_fp.close()

    # SECTION: Print evaluation
    for k, v in sorted(score_dict.items()):
        print(f"{k}: {np.mean(v).item()}")
    
    print("=================================")
    
    return outputs, score_dict # outputs, reviews, score_dict


if __name__ == "__main__":
    # SECTION: Run model on VQA
    
    # run_vqa_evaluation(None, "gpt4")
    # raise Exception()
    hi = 32
    
    start_layers = [12, 14, 16, 18, 20,]
    num_samples = [16, 32, 64, 128, 256, 512]
    model_kwargs_dict = [("baseline", {"mask_layers": range(hi, hi + 1), "num_sample": 64,})] + [
        (f"L[{lo}:{hi}]_S{num_sample}_OFPS", {"mask_layers": range(lo, hi), "num_sample": num_sample,})
        for lo in start_layers for num_sample in num_samples
    ]
    
    plt.rcParams["figure.figsize"] = (10.0, 5.0,)    
    COLORS = np.array([
        [127, 113, 240],
        [247, 214, 124],
        [76, 186, 182],
        [245, 154, 110],
        [217, 17, 17],
        [240, 127, 189],
        [127, 242, 107],
        [237, 92, 208],
    ], dtype=float) / 255
    
    cdict = {hi: "black", **{l: COLORS[i] for i, l in enumerate(start_layers)}}
    print(cdict)
    for model_name, model_kwargs in model_kwargs_dict:
        model = LlavaNextNystromCompressionViT("fps", **model_kwargs)
        # outputs, reviews, score_dict = run_vqa_evaluation(model, model_name)
        outputs, score_dict = run_vqa_evaluation(model, model_name)
        
        ldict = dict(zip(outputs[0].keys(), zip(*map(dict.values, outputs))))
        y, x = torch.tensor(ldict["time"], dtype=torch.float32), torch.tensor(ldict["num_generated_tokens"], dtype=torch.float32)
        mask = (x != 1024)
        x, y = x[mask], y[mask]

        X = torch.stack((x, torch.ones_like(x)), dim=-1)
        lstsq = torch.linalg.pinv(X) @ y
        xl = torch.linspace(-0.1, 1.1, 100) * (x.max() - x.min()) + x.min()
        yl = lstsq[0] * xl + lstsq[1]

        # ax_t.scatter(
        #     x.numpy(force=True), y.numpy(force=True),
        #     s=4, label=f"{model_name}: $y = {lstsq[0].item():.4f}x + {lstsq[1].item():.4f}$, $s = {np.mean(score_dict['f1']).item():.3f}$",
        # )
        # ax_t.plot(
        #     xl.numpy(force=True), yl.numpy(force=True),
        #     color="black",
        #     linewidth=2.0 if model_name == "baseline" else 0.5,
        #     linestyle="--",
        #     zorder=1212.0 if model_name == "baseline" else 0.0,
        # )
        num_sample, mask_layers = model_kwargs["num_sample"], model_kwargs["mask_layers"]
        
        p = (1 / lstsq[0].numpy(force=True), np.mean(score_dict["f1"]).item())
        plt.scatter(
            *p,
            s=16 * (num_sample ** 0.7),
            color=cdict[mask_layers[0]],
            alpha=1.0 if model_name == "baseline" else 0.4,
            label=model_name if model_name == "baseline" else (f"layers[{mask_layers[0]}:32]" if num_sample == 64 else None),
        )
        # ax_s.text(*p, model_name, fontsize=6, ha="left", va="bottom",)
        
        del model
        utils.empty_cache()
    
    # ax_t.set_xlabel("num_generated_tokens")
    # ax_t.set_ylabel("time (s)")
    # ax_t.legend(fontsize=6)

    plt.title("VQA BERT F1 Score by Masked Layers and Number of Samples", fontsize=16,)
    plt.xlabel("Generated Tokens / s", fontsize=14,)
    plt.ylabel("BERT F1 Score", fontsize=14,)
    plt.ylim(top=0.905)
    legend = plt.legend(fontsize=14, loc="lower left",)
    for handle in legend.legend_handles:
        handle._sizes = [128]
    
    plt.savefig("llava_vqa_bert_f1.pdf", bbox_inches="tight",)
    plt.show()






    # hi = 32
    # for lo in range(0, hi, 2):
    #     for num_sample in [1 << k for k in range(4, 11)]:
    #         model = LlavaNextNystromCompressionViT("fps", mask_layers=range(lo, hi))
    #         output_fname, review_fname = run_vqa_evaluation(model, f"layers[{lo}:{hi}]_sample{num_sample}")
    #         del model
    #         utils.empty_cache()

# %%
