import openai
import time
import json
import pickle
from tqdm import tqdm

import re
from datasets import load_dataset
import os
from openai import OpenAI
import pandas
import pickle

def format_judge_inputs(agent1_outputs, agent2_outputs, template, random_indices, selected_data):
    batch_requests = []

    for i, (original_idx, train_data_point) in enumerate(zip(random_indices, selected_data)):
        prompt = f"Question: {train_data_point['ctx']}\n"
        for opt_idx, option in enumerate(train_data_point['endings']):
            prompt += f"{opt_idx}: {option}\n"

        cur_agent1_outputs = agent1_outputs[i]
        cur_agent2_outputs = agent2_outputs[i]

        prompt += "Agent 1 Response: \n"
        prompt += cur_agent1_outputs
        prompt += "Agent 2 Response: \n"
        prompt += cur_agent2_outputs

        batch_requests.append({
            "custom_id": str(original_idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.8,
                "messages": [
                    {"role": "system", "content": template},
                    {"role": "user", "content": prompt}
                ]
            }
        })

    return batch_requests

def create_input(outputs_into_inputs, template, random_indices, selected_data):
    batch_requests = []

    for i, (original_idx, train_data_point) in enumerate(zip(random_indices, selected_data)):
        prompt = f"Question: {train_data_point['ctx']}\n"
        for opt_idx, option in enumerate(train_data_point['endings']):
            prompt += f"{opt_idx}: {option}\n"

        if outputs_into_inputs is not None:
            prompt += "The response you are considering is this: \n"
            prompt += outputs_into_inputs[i]

        batch_requests.append({
            "custom_id": str(original_idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.8,
                "messages": [
                    {"role": "system", "content": template},
                    {"role": "user", "content": prompt}
                ]
            }
        })

    return batch_requests


def extract_outputs(batch_id, client):
    while True:
        status = client.batches.retrieve(batch_id).status
        print(f"Batch status: {status}")
        if status in ["completed", "failed", "expired"]:
            break
        time.sleep(20) 

    
    batch = client.batches.retrieve(batch_id)
    print("-------------------------------")
    print(batch)
    print("-------------------------------")
    output_file_id = batch.output_file_id

    file_response = client.files.content(output_file_id)
    output_data = file_response.text

    json_lines = output_data.strip().splitlines()


    new_inputs = []
    for line in json_lines:
        data = json.loads(line) 
        model_response = data["response"]["body"]["choices"][0]["message"]["content"]
        new_inputs.append(model_response)

    return new_inputs


def get_answer(batch_id, client, ds):
    while True:
        status = client.batches.retrieve(batch_id).status
        print(f"Batch status: {status}")
        if status in ["completed", "failed", "expired"]:
            break
        time.sleep(20)

    batch = client.batches.retrieve(batch_id)
    output_file_id = batch.output_file_id

    file_response = client.files.content(output_file_id)
    output_data = file_response.text

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

    return (none_indices, missclassified_original_indices, wrong_answer, total_examples, output_data)


def extract_label(output_text):
    match = re.search(r'ANSWER:\s*(\d+)', output_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None
    