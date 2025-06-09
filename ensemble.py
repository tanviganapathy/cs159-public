import json
import re
from collections import defaultdict, Counter
from datasets import load_dataset
from dependencies import *
import openai

our_key = 'insert key'
client = openai.OpenAI(api_key=our_key)

batch_ids = [
    "batch_683293eb11f481909bbcf76b559eef70", #metal .7825 --
    "batch_683297f6662c81908b1697644b07d107", #metal .7675 --
    "batch_683298b9ada0819081b6cb5871db9b98",  # metal .77 --
    "batch_6832a1df8b58819096e94a6a9b6e6c9f", #story .7675 --
    "batch_68329f016c2081908d70387c39c7682a", #story .765 --
    # "batch_6832aa7e8f9c819095f06c994f3d57e8", #story .7875
    # "batch_6832a257caec81909c2cbdcb5a9367c6", #ishi .725 
    # "batch_6832a30e12e48190a276fd3d608bbcf6", #ishi .7475
    "batch_6832a258cccc8190bf6f5b41f5642dab", #ishi .7325 --
    # "batch_6832b7da3f3c8190b51e301c48ffde04", #rnrrr .743 
    # "batch_6832b8246ba4819095cc60f3a2a3ec0d", #rnrrr .759
    # "batch_6832b818ce78819089b95f208a86d3b3", #rnrrr 1 none .749
    # "batch_6832bbe673708190911059d6687069be", #poelo .7625
    # "batch_6832b9386ed08190a572375637f49883", #poelo .7725
    # "batch_6832be52af808190b03cff2901ddf8db" #poelo .7675
]

# Dataset
ds = load_dataset("Rowan/hellaswag")

# extract label from model output
def extract_label(output_text):
    match = re.search(r'ANSWER:\s*<?(\d+)>?(?:\s*\*\*)?', output_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None

# check for tie
def has_clear_majority(counter):
    most_common = counter.most_common(2)
    return len(most_common) == 1 or most_common[0][1] > most_common[1][1]

# Collect all answers from all batches
answers_by_question = defaultdict(list)

for batch_id in batch_ids:
    batch = client.batches.retrieve(batch_id)
    output_file_id = batch.output_file_id

    file_response = client.files.content(output_file_id)
    output_data = file_response.text

    # Save raw output if needed
    with open(f"{batch_id}_raw_output.json", "w") as f:
        f.write(output_data)

    for line in output_data.strip().splitlines():
        item = json.loads(line)
        idx = int(item["custom_id"])
        content = item["response"]["body"]["choices"][0]["message"]["content"]
        label = extract_label(content)
        if label is not None:
            answers_by_question[idx].append(label)

# Evaluate majority-voted answers
missclassified_original_indices = []
wrong_answer = []
none_indices = []
total_examples = 0

for idx, predictions in answers_by_question.items():
    if not predictions:
        none_indices.append(idx)
        continue

    counter = Counter(predictions)

    # Reject tied cases
    if not has_clear_majority(counter):
        none_indices.append(idx)
        continue

    majority_label = counter.most_common(1)[0][0]
    gt_label = str(ds['train'][idx]['label'])

    total_examples += 1

    if majority_label != gt_label:
        missclassified_original_indices.append(idx)
        wrong_answer.append(majority_label)
        print(f"OG INDEX: {idx} WRONG ANSWER: {majority_label} CORRECT ANSWER: {gt_label}")

# Final outputs
print("None indices (no answer or tie):", none_indices)
print("Misclassified indices:", sorted(missclassified_original_indices))
print("Wrong answers:", wrong_answer)
print("Total evaluated examples:", total_examples)

if total_examples == 0:
    print("No valid predictions were made.")
else:
    print("Accuracy:", 1 - len(wrong_answer) / total_examples)

