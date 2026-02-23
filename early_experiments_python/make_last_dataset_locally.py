# After so many attempts blocked by running out of RAM in colab and compatibility issues in kaggle,
# I decided to do more data operations locally
# So here I will generate a final data structure which includes:
# {
#                 'taco_question': taco_item['question'],
#                 'ocr_user_code': item['solution'],
#                 'ocr_qwq_critique': item['qwq_critique'],
#                 'taco_soln':random_soln,
#                 'inputs': inputs,
#                 'outputs': outputs,
#                 'judgement': item['judgement'],
#                 'debug_split': item['split'],
#                 'debug_index': item['index']
#             }


# Update: I decided to drop the "test" split of taco. Just sticking to the "test" split to speed thinigs up and reduce complications
# Decided to just use 4000 samples

from tqdm import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import login
import csv
import json
import pandas as pd
import gc
import random
import traceback

# Hide this secret in an env variable or something.
login(token="")

labels = ["nvidia/OpenCodeReasoning-2"]
data_dirs = [None]
splits = ["python"]
idx = 0
label = labels[idx]
ds = load_dataset(label, data_dir = data_dirs[idx], cache_dir="./datasets",  split=splits[idx], streaming=True)

filtered = ds.filter(
    # lambda x: x["judgement"] == 'right' and x["difficulty"]=='EASY')
    lambda x: x["difficulty"]=='EASY')


taco_train = load_dataset(
    "BAAI/TACO",
    split="train",
    streaming=False, # TACO is not compatible with streaming for some reason. It throws an error because it tries to open a URL as a local file
    trust_remote_code=True # TACO requires this to be true
)


train_questions = pd.read_csv('questions_train.csv')

train_indices = train_questions['index'].tolist()

del train_questions
gc.collect()


taco_train0 = taco_train.select(train_indices)
del taco_train
gc.collect()

taco_train_dict = dict(zip(train_indices, taco_train0))


max_samples = 4000
example_points = []
skip = set()
iterator = iter(filtered)
try:
    while len(example_points) < max_samples:
        item = next(iterator)
        if (item['split'],item['index']) in skip:
            continue

        taco_item = None
        if item['split'] != 'train':
            continue
        
        taco_item = taco_train_dict[int(item['index'])]

        input_output = json.loads(taco_item['input_output'])
        inputs = input_output['inputs']
        outputs = input_output['outputs']

        try:
            solutions = json.loads(taco_item['solutions'])
        except ValueError:
            # print("JSONDecodeError on split ",item['split']," index ",item['index'])
            skip.add((item['split'],item['index']))
            continue

        if solutions is None or solutions == []:
            # print("Warning: No solutions for split ",item['split']," index ",item['index'])
            skip.add((item['split'], item['index']))
            continue
        random_soln = random.choice(solutions)

        example_points.append({
                'taco_question': taco_item['question'],
                'ocr_user_code': item['solution'],
                'ocr_qwq_critique': item['qwq_critique'],
                'taco_soln':random_soln,
                'inputs': inputs,
                'outputs': outputs,
                'judgement': item['judgement'],
                'debug_split': item['split'],
                'debug_index': item['index']
            })
except Exception as e:
    print("Error at split: ",item['split']," index:",item['index'], " skipping.")
    # traceback.print_exc()

# export dataset as json
json.dump(example_points, open("dataset.json", "w"), indent=4)

# export dataset as csv
with open("dataset.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=example_points[0].keys(), quoting=csv.QUOTE_ALL)
            # quoting=csv.QUOTE_ALL ensures multi-line fields are properly handled
    writer.writeheader()
    writer.writerows(example_points)