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
import pandas as pd
#from openai.error import RateLimitError

import openai

import json

pickle_save_file = '' # pickle file saved
output_json = '' # output json

our_key = # add api key please
client = openai.OpenAI(api_key=our_key)
batch_id = # batch id

# STEP 3: Poll for status
while True:
    status = client.batches.retrieve(batch_id).status
    print(f"Batch status: {status}")
    if status in ["completed", "failed", "expired"]:
        break

def extract_label(output_text):
    match = re.search(r'ANSWER:\s*<?(\d+)>?(?:\s*\*\*)?', output_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None
    
ds = load_dataset("Rowan/hellaswag")

batch = client.batches.retrieve(batch_id)
output_file_id = batch.output_file_id
error_file_id = batch.error_file_id

file_response = client.files.content(error_file_id)
output_data = file_response.text

# STEP 4: Download results
file_response = client.files.content(output_file_id)
output_data = file_response.text

# Save it to a JSON file
with open(output_json, "w") as f:
    json.dump(output_data, f, indent=4)

missclassified_original_indices = []
wrong_answer = []
none_indices = []

total_examples = 0

json_lines = output_data.strip().splitlines()

all_indices = []
all_e = 0 

for item in json_lines:
    line = json.loads(item)
    original_idx = int(line["custom_id"])
    gpt_answer = None
    response_content = line["response"]["body"]["choices"][0]["message"]["content"]

    if len(response_content) == 1:
        gpt_answer = int(response_content)
    else:
        gpt_answer = extract_label(response_content)
        
    all_indices.append(original_idx)
    all_e += 1 

    total_examples += 1
    if gpt_answer is None:
        total_examples -= 1
        none_indices.append(original_idx)
        continue

    gt_label = int(ds['train'][original_idx]['label'])
    if int(gpt_answer) != gt_label:
        missclassified_original_indices.append(original_idx)
        wrong_answer.append(gpt_answer)
        print(f"OG INDEX: {original_idx} WRONG ANSWER: {gpt_answer} CORRECT ANSWER: {gt_label}")


print("NONE INDICES")
print(none_indices)
print("MISSCLASSIFIED INDICES")
print(sorted(missclassified_original_indices))
print("WRONG ANSWER")
print(wrong_answer)
print(total_examples)
print(len(all_indices))

print("Accuracy:", 1 - ((len(wrong_answer)) / total_examples))

# Save results
data = (missclassified_original_indices, wrong_answer, none_indices)
with open(pickle_save_file, 'wb') as f:
    pickle.dump(data, f)


