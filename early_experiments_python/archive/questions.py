import os
# os.environ['HF_HOME'] = 'D:/huggingface_cache'

# SOMEHOW THIS CODE ONLY FINDS EASY QUESTIONS IN TACO DATASET AND NONE IN THE OTHER 3. NEED TO CHECK
# Verifying with a separate code
# Yes verified. Taco is the only dataset with easy questions in it. Going forward I will ignore the other two datasets.
# {'taco_train': 138665, 'taco_test': 4306}

from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
import csv

login(token="")


easy_python_items = {
    # This will be a key-value store where key is (dataset_name, split_name, index) and value is the entire row item
}

ext_dataset_item_indices = {
    "taco": {
        "train": [],
        "test": [],
    },
    "apps": {
        "train": [],
        "test": [],
    },
    "code_contests": {
        "train": [],
        "test": [],
    },
    "open-r1/codeforces": {
        "train": [],
        "test": [],
    },
}


ext_dataset_hf_names = {
    "taco": "BAAI/TACO",
    "apps": "codeparrot/apps",
    "code_contests": "deepmind/code_contests",
    "open-r1/codeforces": "open-r1/codeforces"
}
# taco and apps above use trust_remote_code=True when loading in the example snippet in the OCR2 dataset page. So keep that in mind while using load_dataset on them
# https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2

# ocr2_dataset = load_dataset("nvidia/OpenCodeReasoning-2", split="python") # his creates a bleddy "Arrow" dataset that is heavy af. Instead imma try streaming it.

ocr2_stream = load_dataset("nvidia/OpenCodeReasoning-2", split="python", streaming=True) 

# Filter for difficulty "Easy" (python is already selected in the split parameter above)


#debug limit 1000
counter = 0
for ocr2_ds_item in tqdm(ocr2_stream, desc="Scanning OCR2 shards"):
    if ocr2_ds_item["difficulty"] != "EASY":
        continue

    key = (ocr2_ds_item["dataset"], ocr2_ds_item["split"], int(ocr2_ds_item["index"]))
    easy_python_items[key] = ocr2_ds_item
    ext_dataset_item_indices[ocr2_ds_item["dataset"]][ocr2_ds_item['split']].append(int(ocr2_ds_item["index"]))

    # #debug
    # counter += 1
    # if counter > 100:
    #     break

print()

# Now for each external dataset, we have indices and splitname. We just need to stream again foreach of them and collect questions
# wherever we have a match in the indices list


# Redo the below separately for all 4 external datasets

# TACO train split:
hf_dataset_name = ext_dataset_hf_names["taco"]
ext_stream = load_dataset(
    hf_dataset_name,
    split="train",
    streaming=False, # TACO is not compatible with streaming for some reason. It throws an error because it tries to open a URL as a local file in windows
    trust_remote_code=True # TACO requires this to be true
)
ext_ds = ext_stream.select(ext_dataset_item_indices["taco"]["train"])
for (index, ext_ds_item) in tqdm(zip(ext_dataset_item_indices["taco"]["train"], ext_ds), desc="Scanning items from \"taco\" dataset train split"):
    query_key = ("taco", "train", index)
    easy_python_items[query_key]["question"] = ext_ds_item["question"]
# TACO test split:
ext_stream = load_dataset(
    hf_dataset_name,
    split="test",
    streaming=False, # TACO is not compatible with streaming for some reason. It throws an error because it tries to open a URL as a local file in windows
    trust_remote_code=True # TACO requires this to be true
)
ext_ds = ext_stream.select(ext_dataset_item_indices["taco"]["test"])
for (index, ext_ds_item) in tqdm(zip(ext_dataset_item_indices["taco"]["test"], ext_ds), desc="Scanning items from \"taco\" dataset test split"):
    query_key = ("taco", "test", index)
    easy_python_items[query_key]["question"] = ext_ds_item["question"]




# Apps train split:
hf_dataset_name = ext_dataset_hf_names["apps"]
ext_stream = load_dataset(
    hf_dataset_name,
    split="train",
    streaming=True,
    trust_remote_code=True # apps requires this to be true
)
# We are using streaming here to save on storage quota in the kaggle space. This does not work well with select() method
# There is no way in huggingface to do stream + select right now. So we have to iterate manually and filter it out.
indices_to_fetch = sorted(ext_dataset_item_indices["apps"]["train"])
if indices_to_fetch:
    for index, ext_ds_item in tqdm(enumerate(ext_stream), desc="Scanning items from the \"apps\" dataset train split"):
        if index in indices_to_fetch:
            query_key = ("apps", "train", index)
            easy_python_items[query_key]["question"] = ext_ds_item["question"]
        if index > indices_to_fetch[-1]:
            break
# Apps test split:
indices_to_fetch = sorted(ext_dataset_item_indices["apps"]["test"])
if indices_to_fetch:
    for index, ext_ds_item in tqdm(enumerate(ext_stream), desc="Scanning items from the \"apps\" dataset test split"):
        if index in indices_to_fetch:
            query_key = ("apps", "test", index)
            easy_python_items[query_key]["question"] = ext_ds_item["question"]
        if index > indices_to_fetch[-1]:
            break





# code_contests train split:
hf_dataset_name = ext_dataset_hf_names["code_contests"]
ext_stream = load_dataset(
    hf_dataset_name,
    split="train",
    streaming=True
)
indices_to_fetch = sorted(ext_dataset_item_indices["code_contests"]["train"])
if indices_to_fetch:
    for index, ext_ds_item in tqdm(enumerate(ext_stream), desc="Scanning items from the \"code_contests\" dataset train split"):
        if index in indices_to_fetch:
            query_key = ("code_contests", "train", index)
            easy_python_items[query_key]["question"] = ext_ds_item["question"]
        if index > indices_to_fetch[-1]:
            break
# code_contests test split:
indices_to_fetch = sorted(ext_dataset_item_indices["code_contests"]["test"])
if indices_to_fetch:
    for index, ext_ds_item in tqdm(enumerate(ext_stream), desc="Scanning items from the \"code_contests\" dataset test split"):
        if index in indices_to_fetch:
            query_key = ("code_contests", "test", index)
            easy_python_items[query_key]["question"] = ext_ds_item["question"]
        if index > indices_to_fetch[-1]:
            break


# open-r1/codeforces train split:
hf_dataset_name = ext_dataset_hf_names["open-r1/codeforces"]
ext_stream = load_dataset(
    hf_dataset_name,
    split="train",
    streaming=True
)
indices_to_fetch = sorted(ext_dataset_item_indices["open-r1/codeforces"]["train"])
if indices_to_fetch:
    for index, ext_ds_item in tqdm(enumerate(ext_stream), desc="Scanning items from the \"open-r1/codeforces\" dataset train split"):
        if index in indices_to_fetch:
            query_key = ("open-r1/codeforces", "train", index)
            easy_python_items[query_key]["question"] = ext_ds_item["question"]
        if index > indices_to_fetch[-1]:
            break
# open-r1/codeforces test split:
indices_to_fetch = sorted(ext_dataset_item_indices["open-r1/codeforces"]["test"])
if indices_to_fetch:
    for index, ext_ds_item in tqdm(enumerate(ext_stream), desc="Scanning items from the \"open-r1/codeforces\" dataset test split"):
        if index in indices_to_fetch:
            query_key = ("open-r1/codeforces", "test", index)
            easy_python_items[query_key]["question"] = ext_ds_item["question"]
        if index > indices_to_fetch[-1]:
            break


print()