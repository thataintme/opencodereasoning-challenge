# Now we have question banks ready.
# We may proceed to now iterate through each dataset entry in the OCR2 dataset and for every question attach a possible correct answer.
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
import csv
import json

# Hide this secret in an env variable or something.
login(token="")

# For querying we will be using the combo (dataset, split, key, judgement). Dataset is always "taco". Split is either "train" or "test". Index is a number. Judgement is either "right" or "wrong"

# First off for each question I will record ids of correct code:

correct_ans_ids = {
    # key would be (split, index)
}

ocr2_stream = load_dataset("nvidia/OpenCodeReasoning-2", split="python", streaming=True) 
for ocr2_ds_item in tqdm(ocr2_stream, desc="Collecting columns with correct answers in OCR2 dataset"):
    if ocr2_ds_item["difficulty"] != "EASY":
        continue
    if ocr2_ds_item['dataset'] == 'taco' and ocr2_ds_item['judgement'] == 'right':
        key = (ocr2_ds_item['split'], ocr2_ds_item['index'])
        correct_ans_ids.setdefault(key, []).append(ocr2_ds_item['id'])

correct_ans_json_s = {
    f"{split}_{index}": item
    for (split, index), item in correct_ans_ids.items()
}

with open("correct_ans_ids.json", mode="w", encoding="utf-8") as f:
    json.dump(correct_ans_json_s, f)


# Now we have a directory for correct answers. We can use that as a bank for solutions.
# Alternatively we can even just randomly select one answer for each question, but that would be bad
#    because we have:
#         taco_train:  90320
#         taco_test:  2463
#         unique taco_train:  4346
#         unique taco_test:  99
# If we were to select 4k correct answers for 90k occurrences, it is possible to overfit on a specific code.


# Plan from here:
# We have a bank of questions we can query
# We have a bank of correct answers for each question key
# Now we can go to training. We will iterate through the ocr2 dataset.
# We will query our question bank for the question. This alongside the "solution" column of ocr2 dataset acts as the prompt. [The solution is not the actual solution, it is a proposed solution that may or may not be correct, determined by the "judgement" column remember]
# For each matching row on OCR2 (EASY, python), we if the code is correct we just use the qwq_critique and judgement as response
#    If the solution code is wrong, we will go to the correct-answer bank and query to get a list of correct solutions.
#         From there we will randomly select one correct solution and use that as a correct solution


# Update: I just realised I missed something. Go to collect_taco_dataset.py
