from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
import csv
# At later stage I realised they only want erraneous solution codes in input.
# From the SQL query tool for OCR2 dataset, theres only 467 rows with wrong code on "Python and EASY"
# So I am now counting the entire dataset to make sure I am right


login(token="")

ocr2_stream = load_dataset("nvidia/OpenCodeReasoning-2", split="python", streaming=True) 

counts = {
}

indices_wrong = set()

for index, ocr2_ds_item in enumerate(tqdm(ocr2_stream, desc="Counting number of samples in OCR2 dataset with wrong code")):
    if ocr2_ds_item["difficulty"].upper() == "EASY":
        split = ocr2_ds_item["split"]
        judgement = ocr2_ds_item["judgement"]
        ds = ocr2_ds_item['dataset']
        if judgement.lower() == "wrong":
            indices_wrong.add((split,index))
        counts[f"{ds}_{split}_{judgement}"] = counts.get(f"{ds}_{split}_{judgement}", 0) + 1

print(counts)
        
#{'train_right': 122906, 'test_right': 3560, 'train_wrong': 15759, 'test_wrong': 746}
with open("wrong_soln_indices.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "index"])  # header
    writer.writerows(indices_wrong)


# Okay so I was wrong. There's more wrong rows. Anyways moving forward...
