import json
import os
from collections import defaultdict

import numpy as np


if __name__ == "__main__":
    review_file = "playground/data/coco2014_val_qa_eval/qa90_gpt4_review.jsonl"
    scores = defaultdict(list)
    with open(review_file) as f:
        for review_str in f:
            review = json.loads(review_str)
            if "category" in review:
                scores[review["category"]].append(review["tuple"])
                scores["all"].append(review["tuple"])
            else:
                if "tuple" in review:
                    scores["all"].append(review["tuple"])
                else:
                    scores["all"].append(review["score"])
    for k, v in sorted(scores.items()):
        stats = np.asarray(v).mean(0).tolist()
        stats = [round(x, 3) for x in stats]
        # print(k, stats, round(stats[1]/stats[0]*100, 1))
        print(k, round(stats[1]/stats[0]*100, 1), round(stats[0] * 10, 1), round(stats[1] * 10, 1))
    print("=================================")
