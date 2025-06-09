import pickle
import pandas as pd

import openai
import random
import time
import json
import pickle
from tqdm import tqdm

import re
from datasets import load_dataset
import random
import os
import json
import openai
from openai import OpenAI
import pandas
import pickle
import time
#from openai.error import RateLimitError

import openai

# Open the file in read-binary mode
with open('info.pkl', 'rb') as f:
    missclassified_idxs_1, wrong_answers_1, none_idxs_1 = pickle.load(f)

with open('info_v2.pkl', 'rb') as f:
    missclassified_idxs_2, wrong_answers_2, none_idxs_2 = pickle.load(f)

with open('info_o4_mini.pkl', 'rb') as f:
    missclassified_idxs_3, wrong_answers_3, none_idxs_3 = pickle.load(f)

with open('info_v3.pkl', 'rb') as f:
    missclassified_idxs_4, wrong_answers_4, none_idxs_4 = pickle.load(f)

print(f"D1: {len(missclassified_idxs_1)}")
# print(missclassified_idxs_1)
# print(wrong_answers_1)
df_1 = pd.DataFrame({
    "Index": missclassified_idxs_1,
    "Wrong Answer Turbo 1": wrong_answers_1
})

# print(df_1)
# exit()

print(f"D2: {len(missclassified_idxs_2)}")
df_2 = pd.DataFrame({
    "Index": missclassified_idxs_2,
    "Wrong Answer Turbo 2": wrong_answers_2
})
print(f"D3: {len(missclassified_idxs_3)}")
df_3 = pd.DataFrame({
    "Index": missclassified_idxs_3,
    "Wrong Answer o4-mini": wrong_answers_3
})

print(f"D4: {len(missclassified_idxs_4)}")
df_4 = pd.DataFrame({
    "Index": missclassified_idxs_4,
    "Wrong Answer Turbo 3": wrong_answers_4
})

combined_wrong = set(missclassified_idxs_1 + missclassified_idxs_2 + missclassified_idxs_3 + missclassified_idxs_4)
print(f"C: {len(combined_wrong)}")
merged_df = df_1.merge(df_2, on='Index', how='outer').merge(df_3, on='Index', how='outer').merge(df_4, on='Index', how='outer')
print(merged_df)
print("at least 1 got wrong (same as combined):", len(merged_df))
print()

# Convert lists to sets
set1 = set(missclassified_idxs_1)
set2 = set(missclassified_idxs_2)
set3 = set(missclassified_idxs_3)
set4 = set(missclassified_idxs_4)

all_overlapping_elements = set1 & set2 & set3 & set4
print(f"LENGTH OF ALL OVERLAPPING: {len(all_overlapping_elements)}")
print(all_overlapping_elements)
all_wrong = merged_df.dropna(subset=['Wrong Answer Turbo 1', 'Wrong Answer Turbo 2', 'Wrong Answer o4-mini', 'Wrong Answer Turbo 3'])
print(all_wrong)
print("all 4 got wrong:", len(all_wrong))
all_wrong.to_pickle("all_wrong.pkl")

# prompt on the wrong ones

api_key = # private api key

ds = load_dataset("Rowan/hellaswag")
client = openai.OpenAI(api_key=api_key)

def extract_label(output_text):
    match = re.search(r'ANSWER:\s*(\d+)', output_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None

# Prepare dataset
random.seed(42)
random_indices = random.sample(range(len(ds['train'])), 400)
selected_data = ds['train'].select(random_indices)
# selected_data = ds['train'].select(all_overlapping_elements)

# STEP 1: Construct batch requests
batch_requests = []
# for original_idx, train_data_point in zip(all_overlapping_elements, selected_data):
for original_idx, train_data_point in zip(random_indices, selected_data):
    #prompt = "Select number corresponding to the correct answer\n"
    prompt = f"Question: {train_data_point['ctx']}\n"
    for opt_idx, option in enumerate(train_data_point['endings']):
        prompt += f"{opt_idx}: {option} \n"


    template = f'''
    Given the question and four options, identify the most coherent continuation by answering the following:


    - WHO is involved? (Identify the main subjects — person, animal, object)
    - WHAT is happening? (Summarize the current action or event)
    - WHERE is it happening? (Identify the physical setting)
    - WHEN is this happening? (Time of day, phase of activity, sequence)
    - WHY is it happening? (Purpose or intention — implied or stated)


    For each option:
    - Does it involve the same WHO?
    - Is the WHAT a natural continuation of the earlier action?
    - Does the WHERE stay consistent?
    - Does the WHEN make sense in the sequence?
    - Is the WHY reasonable, or does it contradict the scene?


    Eliminate options that introduce new characters, places, or implausible motives.


    Output the number corresponding to the final answer in the format:
    ANSWER: number
    '''



    batch_requests.append({
        "custom_id": str(original_idx),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-3.5-turbo", 
            "messages": [
                {"role": "system", "content": template},
                {"role": "user", "content": prompt}
            ]
        }

    })

import json

cur_input_file = "batch_input_gpt_3.5_5W_6.jsonl"

# Save batch_requests to JSONL file
with open(cur_input_file, "w") as f:
    for request in batch_requests:
        f.write(json.dumps(request) + "\n")

retries = 0
while retries < 15:
    print("Curr retries", retries)
    retries += 1
    batch_input_file = client.files.create(
        file = open(cur_input_file, "rb"), 
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id 

    with open(cur_input_file, "rb") as file:
        response = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

    batch_id = response.id
    print(f"Submitted batch with ID: {batch_id}")

    # STEP 3: Poll for status
    while True:
        status = client.batches.retrieve(batch_id).status
        print(f"Batch status: {status}")
        if status in ["completed", "failed", "expired"]:
            break
        if status in ["in_progress"]:
            retries = 20
            break
        time.sleep(30)

exit()

# Elements unique to each list
unique_to_list1 = set1 - set2
unique_to_list2 = set2 - set1

# Elements common to both lists
overlapping_elements = set1 & set2

# Printing results
print(f"D1: {len(data_list1)}")
print(f"D2: {len(data_list2)}")
print(f"Unique to D1: {len(unique_to_list1)}")
print(f"Unique to D2: {len(unique_to_list2)}")
print(f"Overlap: {len(overlapping_elements)}")
print(f"Total Unique Combined (correct): {len(set1 | set2)}")
print(overlapping_elements)

# data_list1.extend(data_list2)
# print(len(set(data_list1)))