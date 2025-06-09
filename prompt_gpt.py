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
import openai


def extract_label(output_text):
    match = re.search(r'ANSWER:\s*(\d+)', output_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None

ds = load_dataset("Rowan/hellaswag")

print("RANDOM DATASET INFO")
print(len(ds['test']))
print(ds['train'][0]['ctx'])
print(ds['train'][0]['ctx_a'])
print(ds['train'][0]['ctx_b'])
print(ds['train'][0]['endings'])
print(ds['train'][0]['label'])
print(ds['train'][0].keys())
print(len(ds['train']))

our_key = # add api key please
client = OpenAI(api_key = our_key)

def get_openai_response(prompt, template):
    response = client.chat.completions.create(
            #model = "gpt-4",
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": template},
                {"role": "user", "content": prompt}
            ] #set temperature, batch API open AI 
        )
    return response.choices[0].message.content

def extract_label(output_text):
    match = re.search(r'ANSWER:\s*(\d+)', output_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None
    
def safe_openai_call(prompt, template):
    while True:
        try:
            full_response = get_openai_response(prompt, template)
            return full_response
        except openai.RateLimitError as e:
            print("Rate limit error caught! Waiting before retrying...")
            print(f"Error details: {str(e)}")

            # Parse suggested wait time from error message if possible
            suggested_wait = 20  # default 20 seconds if parsing fails
            if hasattr(e, 'response') and e.response:
                try:
                    suggested_wait = int(e.response.json()['error']['message'].split('try again in ')[1].split('s')[0])
                except Exception:
                    pass  # if parsing fails, keep default 20s

            # Add some random jitter to be safe
            wait_time = suggested_wait + 65 + random.uniform(1, 5)
            print(f"Sleeping for {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    
template = '''
Given the mcq, choose 0, 1, 2, or 3 as the number corresponding to the correct answer
Output in the format:
ANSWER: number
'''

random.seed(42)
random_indices = random.sample(range(len(ds['train'])), 400)

selected_data = ds['train'].select(random_indices)

# for selected_idx, (original_idx, train_data_point) in enumerate(zip(random_indices, selected_data)):
#     print(f"{original_idx} ---------------------------")
#     print("Select number corresponding to the correct answer")
#     print(f"Question: {train_data_point['ctx']}")
#     for opt_idx, option in enumerate(train_data_point['endings']):
#         print(f"{opt_idx}: {option}")
#     print(f"Answer: {train_data_point['label']}")

missclassified_original_indices = []
wrong_answer = []
total_examples = len(random_indices)
none_indices = []
count = 0
for selected_idx, (original_idx, train_data_point) in enumerate(zip(random_indices, selected_data)):
    if count % 10 == 0:
      print(f"IT HAS MADE PROGRESS OF GOING UP BY 10: {count}")
    count += 1
    print(f"COUNT: {count}")
    prompt = "Select number corresponding to the correct answer\n"
    prompt += f"Question: {train_data_point['ctx']}\n"
    for opt_idx, option in enumerate(train_data_point['endings']):
        prompt += f"{opt_idx}: {option} \n"

    full_response = safe_openai_call(prompt, template)

    gpt_answer = extract_label(full_response)

    print(f"Response: {full_response}")
    print("EXTRACTED ANSWER")
    print(gpt_answer)
    print("------------------")
    print()

    if not gpt_answer:
      total_examples -= 1
      none_indices.append(original_idx)

    if gpt_answer and int(gpt_answer) != int(train_data_point['label']):
      missclassified_original_indices.append(original_idx)
      wrong_answer.append(gpt_answer)

      print(prompt)
      print(f"Wrong Answer: {gpt_answer}")
      print(f"Right Answer: {train_data_point['label']}")

print(1-(len(wrong_answer))/total_examples)

data = (missclassified_original_indices, wrong_answer, none_indices)
with open('info.pkl', 'wb') as f:
  pickle.dump(data, f)


