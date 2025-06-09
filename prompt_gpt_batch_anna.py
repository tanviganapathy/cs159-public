import openai
import random
import time
import json
import pickle
from tqdm import tqdm
import json

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
import pandas as pd

cur_input_file = "batch_input_modifed.jsonl"


df = pd.read_pickle("all_wrong.pkl")
ds = load_dataset("Rowan/hellaswag")

def extract_label(output_text):
    match = re.search(r'ANSWER:\s*(\d+)', output_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None

our_key = '' # add key


client = openai.OpenAI(api_key=our_key)

# template = '''
# Given the mcq, choose 0, 1, 2, or 3 as the number corresponding to the correct answer

# You are an expert problem solver using the IDEAL framework.

# 1. Identify the situation.
# 2. Define the problem.
# 3. Explore possible endings.
# 4. Act: Choose the best one.
# 5. Look back: Justify your choice.

# Don't output the answer until you've completed all 5 steps. At the end, output:
# ANSWER: number

# Make sure to carefully reason and then answer
# '''

# template = '''
# Given the mcq, choose 0, 1, 2, or 3 as the number corresponding to the correct answer

# You are an expert problem solver using the IDEAL framework.

# 1. Identify the situation.
# 2. Define the problem.
# 3. Explore possible endings.
# 4. Act: Choose the best one.
# 5. Look back: Justify your choice.

# Don't output the answer until you've completed all 5 steps. 
# Explain your thought process through each of the steps. 

# At the end, output:
# ANSWER: number

# Make sure to carefully reason and then answer
# '''

# template = '''
# Use both intuitive and analytical thinking (System 1 and System 2) to solve the problem:


# 1. Initial Reaction: What is your first intuitive response?
# 2. Deliberate Analysis: Slow down. Analyze the choices carefully—what logic or world knowledge applies?
# 3. Compare the two responses: Are they the same? If not, which is more reliable and why?
# 4. Final Decision: Choose the answer based on your full reasoning.
# 5. Reflection: Was intuition misleading or helpful? What does this say about how you reason?


# Then output:
# ANSWER: number
# '''

# template = '''
# Use both intuitive and analytical thinking (System 1 and System 2) to solve the problem:


# 1. Initial Reaction: What is your first intuitive response?
# 2. Deliberate Analysis: Slow down. Analyze the choices carefully—what logic or world knowledge applies?
# 3. Compare the two responses: Are they the same? If not, which is more reliable and why?
# 4. Final Decision: Choose the answer based on your full reasoning.
# 5. Reflection: Was intuition misleading or helpful? What does this say about how you reason?

# MAKE SURE TO ANSWER IN THE FOLLOWING FORMAT 

# Then output:
# ANSWER: number

# Example:
# ANSWER: 0
# '''

# template = '''
# You are an expert problem solver using the IDEAL framework:

# Identify the situation.

# Define the problem.

# Explore possible endings.

# Act: Choose the best one.

# Look back: Justify your choice.

# Given a multiple-choice question (MCQ), select the number (0, 1, 2, or 3) corresponding to the correct answer.
# Do not output the answer until you've completed all 5 steps.
# Explain your thought process clearly at each step.
# End with: ANSWER: number

# Example
# Question:
# People ride horses in the shallow waters of the sea. People ride the horses in the water along the shore. A horse defecates in the water.
# people
# 0: form a line with the horses while riding in the ocean.
# 1: swim in the choppy water.
# 2: stand on the horses giving the people water water when children stand on the sides.
# 3: assist the horses in the water.

# Step 1: Identify the situation
# People are riding horses in shallow sea water along the shore. A horse defecates.

# Step 2: Define the problem
# We need to infer what people do next — the most plausible continuation of the story.

# Step 3: Explore possible endings

# 0: Logical but doesn’t respond to the defecation.

# 1: Abrupt change to swimming — less coherent.

# 2: Nonsensical and grammatically broken.

# 3: Plausible response to the horse defecating — maybe the people assist the horses due to distress or concern.

# Step 4: Act
# Choose option 3, as it is the only one that reasonably and coherently responds to the situation.

# Step 5: Look back
# Option 3 fits best: it maintains the narrative, relates to the defecation event, and implies a caretaking or corrective action.

# ANSWER: 3

# Now solve the following questions using the same IDEAL framework reasoning.
# '''

# METAL 
template = '''
You are a thoughtful reasoner using the METAL framework:

M: Make a claim — state which answer you think is correct.  
E: Provide evidence — explain why that answer fits best.  
T: Think of a counter — consider another choice that might seem plausible.  
A: Acknowledge limitations — note any uncertainty or assumptions.  
L: Learn from it — reflect on what makes the chosen answer better overall.

Given a multiple-choice question (MCQ), select the number (0, 1, 2, or 3) corresponding to the correct answer.  
Do not output the answer until all 5 steps are completed.  
End with: ANSWER: number

Example 1:
Question: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then  
Options:  
0: the man adds wax to the windshield and cuts it.  
1: a person boards a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.  
2: the man puts on a Christmas coat, knitted with netting.  
3: the man continues removing the snow on his car.

M: I claim that option 3 is correct.  
E: The man is already interacting with snow on a car, so continuing to remove it is a logical next step.  
T: Option 2 might seem plausible since it's winter and he might put on winter gear.  
A: However, there's no prior mention of needing new clothes or a costume change — and option 2 lacks relevance.  
L: The clearest progression is from writing in snow to removing it, making 3 the most natural continuation.  
ANSWER: 3

Example 2:
Question: He takes his lighter and lights the newspaper in several places to start the fire. The bonfire starts burning and continues to burn. he  
Options:  
0: plays with the dog and makes two cookies.  
1: adds a few more twigs to keep the flames burning.  
2: gets up and attempts to put a flag on it, fails and makes a complete ass out of himself.  
3: puts on equipment and stools.

M: I claim that option 1 is correct.  
E: Adding twigs is a reasonable action when tending a new fire to sustain it.  
T: Option 2 might seem like it adds drama or humor, but it veers into implausibility.  
A: While we don’t know his full intent, the narrative tone seems functional rather than chaotic or comedic.  
L: Option 1 stays focused on maintaining the fire, which is the core of the scene.  
ANSWER: 1

Example 3:
Question: The roof is done and a view of the entire house is shown to show off the finished roof. the woman  
Options:  
0: is standing in front of the home, smiling while talking.  
1: interviews the man again and leaves the room.  
2: shows the soil with two lay-ups of shingle and applies a layer onto the top shingle.  
3: stacks the bags on the side and begins putting stencils on the top.

M: I claim that option 0 is correct.  
E: Showing off a finished roof followed by the woman smiling in front of the house aligns with a project wrap-up moment.  
T: Option 2 might be tempting if the roof weren’t already declared “done,” suggesting more work.  
A: It’s possible the video could jump back in time, but that’s unlikely given the “roof is done” framing.  
L: Option 0 matches the narrative closure style often used in documentary or instructional formats.  
ANSWER: 0
'''

def extract_label(output_text):
    match = re.search(r'ANSWER:\s*(\d+)', output_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None

# Prepare dataset
random.seed(42)
missclassified_indices = df['Index']
selected_data = ds['train'].select(missclassified_indices)

# if 0 in missclassified_indices:
#     print("YES 0")

# if 1 in missclassified_indices:
#     print("YES 1")

# if 2 in missclassified_indices:
#     print("YES 2")

# exit()


# STEP 1: Construct batch requests
batch_requests = []
for original_idx, train_data_point in zip(missclassified_indices, selected_data):
    prompt = f"Question: {train_data_point['ctx']}\n"
    for opt_idx, option in enumerate(train_data_point['endings']):
        prompt += f"{opt_idx}: {option} \n"


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

# Save batch_requests to JSONL file
with open(cur_input_file, "w") as f:
    for request in batch_requests:
        f.write(json.dumps(request) + "\n")

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
    time.sleep(3000)
