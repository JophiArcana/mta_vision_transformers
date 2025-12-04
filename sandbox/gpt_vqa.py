import json
import time
from tqdm import tqdm
from typing import List

import openai
from openai import OpenAI

client = OpenAI()


def get_eval(content: str, max_tokens: int) -> str:
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
        except openai.RateLimitError:
            pass
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response.choices[0].message.content

def parse_score(review: str) -> List[int]:
    score_pair = review.split("\n")[0]
    score_pair = score_pair.replace(",", " ").split(" ")
    assert len(score_pair) == 2, f"Number of scores should be 2 but got {2}"
    return [*map(float, score_pair)]


if __name__ == "__main__":
    lo, hi = 16, 24

    rootdir = "playground/data/coco2014_val_qa_eval"
    question_fp = open(f"{rootdir}/qa90_questions.jsonl")
    target_fp = open(f"{rootdir}/qa90_gpt4_answer.jsonl")
    input_fp = open(f"{rootdir}/qa90_our_answer_{lo}:{hi}.jsonl")

    eval_rootdir = "llava/eval/table"
    rule_dict = json.load(open(f"{eval_rootdir}/rule.json", "r"))

    context_list = [json.loads(line) for line in open(f"{eval_rootdir}/caps_boxes_coco2014_val_80.jsonl")]
    image_to_context = {context["image"]: context for context in context_list}

    with open(f"{rootdir}/qa90_gpt4_review.json", "a") as review_fp:
        for idx, line in tqdm(enumerate(zip(question_fp, target_fp, input_fp))):
            ques, ans1, ans2 = map(json.loads, line)
            
            inst = image_to_context[ques["image"]]
            cap_str = "\n".join(inst["captions"])
            box_str = "\n".join([f"{instance['category']}: {instance['bbox']}" for instance in inst['instances']])

            category = ques["category"]
            rule = rule_dict[category]
            prompt = rule["prompt"]
            role = rule["role"]
            
            content = (f"[Context]\n{cap_str}\n\n{box_str}\n\n"
                    f"[Question]\n{ques['text']}\n\n"
                    f"[{role} 1]\n{ans1['text']}\n\n[End of {role} 1]\n\n"
                    f"[{role} 2]\n{ans2['text']}\n\n[End of {role} 2]\n\n"
                    f"[System]\n{prompt}\n\n")
        
            review = get_eval(content, 1024)
            scores = parse_score(review)
            review_fp.write(json.dumps({
                "id": idx,
                "question_id": ques["question_id"],
                "category": category,
                "content": review,
                "scores": scores
            }) + "\n")
            review_fp.flush()

        review_fp.close()
