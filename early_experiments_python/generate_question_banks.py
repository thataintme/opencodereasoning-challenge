# To catch up, there are "easy" python questions only in the train and test splits of the Taco dataset.
# Flow for this script:
# 1. Stream through the ocr2 dataset and filter for "easy" questions in the "python" split.
# 2. Store the indices and questions for these items in a set
# 3. Load the Taco dataset and for each index we filtered, get the corresponding question and store to the csv file
# 4. Save the dictionary as a file for later use.

# [Remember that we don'y worry about dataset name because only Taco has easy python questions. Check the archived old script for more details on this discovery.]

# I later realised there's a question_id which we may find useful... we can try to use that later
# Never mind there is no way to us it in the external dataset

from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
import csv
import json

# Hide this secret in an env variable or something.
login(token="")

taco_train_indices = set()
taco_test_indices = set()


ocr2_stream = load_dataset("nvidia/OpenCodeReasoning-2", split="python", streaming=True) 
# Filter for difficulty "Easy" (python is already selected in the split parameter above)
for ocr2_ds_item in tqdm(ocr2_stream, desc="Scanning OCR2 shards"):
    if ocr2_ds_item["difficulty"] != "EASY":
        continue
    if ocr2_ds_item["split"] == "train":
        taco_train_indices.add(int(ocr2_ds_item["index"]))
    elif ocr2_ds_item["split"] == "test":
        taco_test_indices.add(int(ocr2_ds_item["index"]))
    else:
        print("Warning: unrecognised split: ", ocr2_ds_item["split"]," at index: ", ocr2_ds_item["index"])
train_indices = list(taco_train_indices)
test_indices = list(taco_test_indices)

# Debug: storing in a json so we dont need to iterate again through this dataset next time (saving time)
with open("taco_indices.json", "w") as f:
    json.dump({
        "train": sorted(taco_train_indices),
        "test": sorted(taco_test_indices)
    }, f)
# Debug: In case we are using the above created json file, use this to retrieve the indices list:
# indices = json.load(open("taco_indices.json"))
# train_indices, test_indices = indices['train'], indices['test']



# Now we have the indices of all questions we want.
# Next we load the Taco dataset and extract these questions.

# TACO train split:
ext_stream_train = load_dataset(
    "BAAI/TACO",
    split="train",
    streaming=False, # TACO is not compatible with streaming for some reason. It throws an error because it tries to open a URL as a local file
    trust_remote_code=True # TACO requires this to be true
)
countertrain = 0
ext_ds_train = ext_stream_train.select(train_indices)
# loop over dataset and write each row immediately (STREAMING THE DATA TO A CSV)
with open("questions_train.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # write header row
    writer.writerow(["index", "question"])
    for index, ext_ds_item in tqdm(zip(train_indices, ext_ds_train), desc='Scanning items from "taco" dataset train split', total=len(train_indices)):
        question = ext_ds_item["question"]
        # write directly to CSV
        writer.writerow([index, question])

# TACO test split:
ext_stream_test = load_dataset(
    "BAAI/TACO",
    split="test",
    streaming=False, # TACO is not compatible with streaming for some reason. It throws an error because it tries to open a URL as a local file
    trust_remote_code=True # TACO requires this to be true
)
countertest = 0
ext_ds_test = ext_stream_test.select(test_indices)
# loop over dataset and write each row immediately (STREAMING THE DATA TO A CSV)
with open("questions_test.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # write header row
    writer.writerow(["index", "question"])
    for index, ext_ds_item in tqdm(zip(test_indices, ext_ds_test), desc='Scanning items from "taco" dataset test split', total=len(test_indices)):
        question = ext_ds_item["question"]
        # write directly to CSV
        writer.writerow([index, question])



# Now we have questions in these two files that may act as the question bank. Instead of attaching these questions to the dataset (which will
#    severely strike memory usage), we are better off leaving the question bank here and querying it whenever needed for finetuning etc.
# Until then, we will treat the (index, split) pair as keys to deal with thisdataset going forward

# proceed to attach_correct_solutions.py