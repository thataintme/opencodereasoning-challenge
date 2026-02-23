from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

login(token="")

ocr2_stream = load_dataset("nvidia/OpenCodeReasoning-2", split="python", streaming=True) 

counts = {
}

items = {

}

indices = set()

for index, ocr2_ds_item in enumerate(tqdm(ocr2_stream, desc="Counting number of easy questions in each of the three datasets and their splits")):
    if ocr2_ds_item["difficulty"].upper() == "EASY":
        dataset_name = ocr2_ds_item["dataset"]
        split = ocr2_ds_item["split"]
        items[index] = ocr2_ds_item
        indices.add(index)
        counts[f"{dataset_name}_{split}"] = counts.get(f"{dataset_name}_{split}", 0) + 1

print(counts)
        