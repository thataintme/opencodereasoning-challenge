from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
import csv

# Hide this secret in an env variable or something.
login(token="")

# Totally available Easy Python questions in the OCR2 dataset:
# {'taco_train': 138665, 'taco_test': 4306}

easy_python_items = {
    # This will be a key-value store where key is (dataset_name, split_name, index) and value is the entire row item
    # Update: It was discovered that only the "Taco" dataset has "Easy" python questions so dataset stays the same.
    # Only split and index act as key
    # Update: It was discovered that some indices repeat so we will use a different flow
}

ext_dataset_item_indices = {
    "taco_test": [],
    "taco_train": []
}

question_indices = {

}

taco_hf_name = "BAAI/TACO"
# taco and apps above use trust_remote_code=True when loading in the example snippet in the OCR2 dataset page. So keep that in mind while using load_dataset on them
# https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2



ocr2_stream = load_dataset("nvidia/OpenCodeReasoning-2", split="python", streaming=True) 

# Filter for difficulty "Easy" (python is already selected in the split parameter above)

for ocr2_ds_item in tqdm(ocr2_stream, desc="Scanning OCR2 shards"):
    if ocr2_ds_item["difficulty"] != "EASY":
        continue

    key = (f"taco_{ocr2_ds_item['split']}", int(ocr2_ds_item["index"]))
    easy_python_items[key] = ocr2_ds_item
    ext_dataset_item_indices[f"taco_{ocr2_ds_item['split']}"].append(int(ocr2_ds_item["index"]))

print("easy_python_items... filtered for difficulty easy. Found ", len(easy_python_items), " items.")
print("ext_dataset_item_indices has lengths: ")
print("taco_train: ", len(ext_dataset_item_indices["taco_train"]))
print("taco_test: ", len(ext_dataset_item_indices["taco_test"]))
print("unique taco_train: ", len(set(ext_dataset_item_indices["taco_train"])))
print("unique taco_test: ", len(set(ext_dataset_item_indices["taco_test"])))
print("=================================================")


# TACO train split:
ext_stream = load_dataset(
    taco_hf_name,
    split="train",
    streaming=False, # TACO is not compatible with streaming for some reason. It throws an error because it tries to open a URL as a local file in windows
    trust_remote_code=True # TACO requires this to be true
)
ext_ds = ext_stream.select(ext_dataset_item_indices["taco_train"])
countertrain = 0
for (index, ext_ds_item) in tqdm(zip(ext_dataset_item_indices["taco_train"], ext_ds), desc="Scanning items from \"taco\" dataset train split"):
    query_key = ("taco_train", index)
    easy_python_items[query_key]["question"] = ext_ds_item["question"]
    countertrain += 1

print("Finished scanning taco train split. Counter: ", countertrain)
print("size of easy_python_items: ", len(easy_python_items))



# TACO test split:
ext_stream = load_dataset(
    taco_hf_name,
    split="test",
    streaming=False, # TACO is not compatible with streaming for some reason. It throws an error because it tries to open a URL as a local file in windows
    trust_remote_code=True # TACO requires this to be true
)
ext_ds = ext_stream.select(ext_dataset_item_indices["taco_test"])
for (index, ext_ds_item) in tqdm(zip(ext_dataset_item_indices["taco_test"], ext_ds), desc="Scanning items from \"taco\" dataset test split"):
    query_key = ("taco_test", index)
    easy_python_items[query_key]["question"] = ext_ds_item["question"]


# Now all questions are added. Ignoring all keys and only writing values to the final CSV file
# Debug: There were only 4449 rows in the final dataset instead of 138665+4306. Need to investigate how they got lost.
# Never mind so basically there's only a few questions that repeat.
# print("taco_train: ", len(ext_dataset_item_indices["taco_train"]))
# print("taco_test: ", len(ext_dataset_item_indices["taco_test"]))
# print("unique taco_train: ", len(set(ext_dataset_item_indices["taco_train"])))
# print("unique taco_test: ", len(set(ext_dataset_item_indices["taco_test"])))
# taco_train:  90320
# taco_test:  2463
# unique taco_train:  4346
# unique taco_test:  99
# This means to make the dataset stronger, we need to try a different key so that we preserve all these occurrences of the same indices

items = list(easy_python_items.values())
with open("easy_python_questions_060226.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = items[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for item in items:
        writer.writerow(item)

print("Finished writing to CSV file")

# Later I found that a number of these questions have <image> tag with no actual image data. So I might have to filter these out
# Not now, I will do this while loading the dataset for training the model (in the kaggle notebook).

# Hmm... I just realised that there is no column for the correct solution code whenever the answer is wrong. I dont know what do to about that
# Maybe generate correct code from a different model? i dont know... Maybe i should track these questions down and collect the best solution?
# Or maybe this is also just innately done by the model?



# Latest update: Found that most indices are repeated multiple times so the actual number of unique questions is only around 4449.
# We will be going over a different flow now. 
# Go to generate_dataset_with_questions.py to see new flow
