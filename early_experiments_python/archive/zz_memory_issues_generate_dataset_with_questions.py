# To catch up, there are "easy" python questions only in the train and test splits of the Taco dataset.
# Flow for this script:
# 1. Stream through the ocr2 dataset and filter for "easy" questions in the "python" split.
# 2. Store the indices and questions for these items as keys in a dictionary (with empty values). Key would be (split, index)
# 3. Load the Taco dataset and for each index we filtered, get the corresponding question and store to the above dictionary
#               - Basically in the old script we were saving the entire row, but now we are only saving questions because we found there are multiple rows per question
# 
# Update: to avoid complication, I decided to save the questions locally first and use it later whenever needed
# 4. Save the dictionary as a file for later use.


from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
import csv

# Hide this secret in an env variable or something.
login(token="")

# Totally available Easy Python questions in the OCR2 dataset:
# {'taco_train': 138665, 'taco_test': 4306}

questions = {
    # This will be a key-value store. Key is (split_name, index)
    # value is the question string corresponding to that entry in the Taco dataset.
    # [Remember that we don'y worry about dataset name because only Taco has easy python questions. Check the archived old script for more details on this discovery.]
}

# ext_dataset_items = [
#     # This will be a list of all rows we extract that match our conditions (Easy difficulty, Python language).
#     # Update: to avoid confusion, this was taken to a separate script.
# ]


ocr2_stream = load_dataset("nvidia/OpenCodeReasoning-2", split="python", streaming=True) 
# Filter for difficulty "Easy" (python is already selected in the split parameter above)


# We stream through the ocr2 dataset and populate both the above variables in one go.
# questions is for storing the question text, ext_dataset_items is to store the entire row entries
for ocr2_ds_item in tqdm(ocr2_stream, desc="Scanning OCR2 shards"):
    if ocr2_ds_item["difficulty"] != "EASY":
        continue

    key = (f"taco_{ocr2_ds_item['split']}", int(ocr2_ds_item["index"]))
    questions[key] = None # Initialise with empty string for now.

    # ext_dataset_items.append(ocr2_ds_item) # Update: to avoid confusion this was taken to a separate script


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
indices_train = list(questions.keys())
ext_ds_train = ext_stream_train.select(indices_train) # get all rows for matching indices)
for (index, ext_ds_item) in tqdm(zip(indices_train, ext_ds_train), desc="Scanning items from \"taco\" dataset train split"):
    query_key = ("taco_train", index)
    questions[query_key] = ext_ds_item["question"]



# TACO test split:
ext_stream_test = load_dataset(
    "BAAI/TACO",
    split="test",
    streaming=False, # TACO is not compatible with streaming for some reason. It throws an error because it tries to open a URL as a local file
    trust_remote_code=True # TACO requires this to be true
)
countertest = 0
indices_test = list(questions.keys())
ext_ds_test = ext_stream_test.select(indices_test) # get all rows for matching indices)
for (index, ext_ds_item) in tqdm(zip(indices_test, ext_ds_test), desc="Scanning items from \"taco\" dataset test split"):
    query_key = ("taco_test", index)
    questions[query_key] = ext_ds_item["question"]


# Now to store the questions dictionary as a csv file for later use.
with open("easy_python_questions.csv", mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["split", "index", "question"]) # Write header
    for key, question in questions.items():
        split, index = key
        writer.writerow([split, index, question])


# out of memory. Go to generate_question_bank.py to see updated version
