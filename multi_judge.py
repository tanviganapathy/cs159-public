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
import pandas as pd
import pickle
import time
#from openai.error import RateLimitError

import openai

import json

api_key = # private api key
client = openai.OpenAI(api_key=api_key)

# batch_id_1 = "batch_6827f1cc832c8190a88b09e3ef4bce9a" # 3.5 turbo thematic flow detection run 6 --> testing double example on all 400, with reasoning, template and prompt are separate
# # batch_id_2 = "batch_68329ea7be108190985ba25fd590dbd7" # 3.5 turbo prepositions 4 --> with reasoning
# batch_id_2 = "batch_6832b4cf48c08190ac1bfebb3118edc3" # Accuracy: 0.73, 3.5 turbo poelo 1 --> with reasoning
# batch_id_3 = "batch_6832ad3ba1148190894254c98a1ec49b" # 3.5 turbo metal 1 --> with reasoning


batch_id_1 = "batch_6827f49eaea88190bdc7a54f76cadf2e" # 3.5 turbo thematic flow detection run 5 --> testing double example on all 400, no reasoning, template and prompt are separate
batch_id_2 = "batch_6832b6be6a648190aea9614a52d7e1a5" # Accuracy: 0.75, 3.5 turbo poelo 2 --> no reasoning
batch_id_3 = "batch_6832aa267c90819082153f4f4c47e097" # 3.5 turbo metal 1 --> no reasoning

    
ds = load_dataset("Rowan/hellaswag")

def get_batch_output_responses(batch_id):
    batch = client.batches.retrieve(batch_id)
    print(batch)
    output_file_id = batch.output_file_id
    error_file_id = batch.error_file_id

    # STEP 4: Download results
    file_response = client.files.content(output_file_id)
    output_data = file_response.text
    return output_data

json_lines1 = get_batch_output_responses(batch_id=batch_id_1).strip().splitlines()
json_lines2 = get_batch_output_responses(batch_id=batch_id_2).strip().splitlines()
json_lines3 = get_batch_output_responses(batch_id=batch_id_3).strip().splitlines()

# Load both into dictionaries mapping custom_id to content
response_dict_1 = {}
response_dict_2 = {}
response_dict_3 = {}

for item in json_lines1:
    line = json.loads(item)
    custom_id = int(line["custom_id"])
    response_context = line["response"]["body"]["choices"][0]["message"]["content"]
    response_dict_1[custom_id] = response_context

for item in json_lines2:
    line = json.loads(item)
    custom_id = int(line["custom_id"])
    response_content = line["response"]["body"]["choices"][0]["message"]["content"]
    response_dict_2[custom_id] = response_content

for item in json_lines3:
    line = json.loads(item)
    custom_id = int(line["custom_id"])
    response_content = line["response"]["body"]["choices"][0]["message"]["content"]
    response_dict_3[custom_id] = response_content

# Create list of rows for the DataFrame
rows = []
for custom_id in response_dict_1:
    if custom_id in response_dict_2 and custom_id in response_dict_3:
        rows.append({
            "original_idx": custom_id,
            "response_content_1": response_dict_1[custom_id],
            "response_content_2": response_dict_2[custom_id],
            "response_content_3": response_dict_3[custom_id],
        })

# Create the DataFrame
df = pd.DataFrame(rows)

print(df.head())

# Prepare dataset
random.seed(42)
random_indices = random.sample(range(len(ds['train'])), 400)
selected_data = ds['train'].select(random_indices)
# selected_data = ds['train'].select(all_overlapping_elements)

template = f'''

You are given a multiple-choice commonsense reasoning question. The task is to identify the most plausible continuation of a given context.

Each of the four choices (0, 1, 2, 3) is a possible continuation. Only one is correct.

You are also given responses from different agents using different prompting strategies. Each response includes a final answer selected by that agent.

Your task is to:
1. Carefully read the context and answer choices.
2. Review the different agent responses and the rationale or answer each provides.
3. Based on the context, answer choices, and agent responses, output the number corresponding to the best final answer (0, 1, 2, or 3) in the format: ANSWER: number
'''

template_1 = f'''
Given the question and choices, choose the number (0, 1, 2, or 3) corresponding to the best continuation of the underlying emotional and thematic flow.

Step 1: 
When reading the question, focus specifically on understanding the following:
1. What is the emotional tone? (for example, happy, tense, casual, tragic)?
2. What is the type of activity? (for example, socializing, competing, relaxing, problem-solving?)
3. What is the bigger theme? (for example, friendship, fun, work stress, danger?)

Step 2:
Then for each of the multiple choice options 0, 1, 2, or 3, consider:
1. Does it continue the emotional tone?
2. Does it fit the same kind of theme or activity? Or, does it feel abrupt, mismatched, or out of the emotional rhythm?

Choose the most realistic and coeherent continuation — it should follow naturally from the scene and stay in the same context. 
Avoid answers that introduce unrelated actions or illogical jumps. When unsure, prefer the option that maintains the underlying emotional and thematic flow.

Output the number corresponding to the final answer in the format:
ANSWER: number

##
Example Question: The woman then stands up and walks to a part of the roof where she lifts up a black shingle on the roof. the woman 
0: climbs up the black shingle and smiles at the camera. 
1: then lifts up another shingle and throws them onto the roof and lights it.
2: is now on standing on the ground in front of the house smiling and talking. 
3: then climbs on front the shingle and hangs from the cord while she talks to another woman who is seated on a second shingle.

Step 1: Predict tone
- Emotional tone: Calm, focused
- Activity: Problem-solving or inspecting
- Theme: Mundane task or setup

Step 2: Evaluate Options
- Option 0 is physically implausible and breaks tone with performative gesture, so reject this answer
- Option 1 is a sudden, dangerous escalation, which is not aligned with calm tone, so reject this answer
- Option 2 is a realistic follow-up — she finishes and casually socializes, so accept this answer
- Option 3 is surreal and absurd — completely breaks realism and tone, so reject this answer

ANSWER: 2

##
Example Question: Two people are seen sitting before a wave pool and one leads another out onto the water on a board. the person 
0: falls when another attempts to ride the board on the water. 
1: continues to play with the girl and herself. 
2: continues pushing and eventually throws off a large wave and they all cheer. 
3: goes over the side and another shows up on the side with the same person holding a dog close to their chest.

Step 1: Predict tone
- Emotional tone: Light, exploratory
- Activity: Trying out a water activity
- Theme: Casual recreation or learning

Step 2: Evaluate Options
- Option 0 is a realistic and natural mishap during a first attempt, fitting the exploratory tone, so accept this answer
- Option 1 is vague and awkwardly phrased, so reject this answer
- Option 2 jumps too far ahead emotionally and disrupts the pacing, so reject this answer
- Option 3 introduces unrelated elements and breaks realism, so reject this answer

ANSWER: 0
'''

template_2 = '''
P: Preview and pick — Read the four options carefully. Which one seems like the most *realistic and natural* continuation of the situation? Pick that first, *explicitly stating why it maintains continuity of action, setting, or character intent*.
O: Outrule one — Identify and eliminate one option that clearly *breaks continuity* (e.g., jumps to a new, unrelated scene), *introduces something implausible or contradictory*, or *drastically alters the established tone*. Explain *why* it fails based on these criteria.
E: Examine another — Look at a tempting but imperfect option. Why is it *less appropriate* than your first choice? Focus on how it *disrupts logical flow, introduces minor inconsistencies, or is less direct/efficient* in continuing the scene.
L: Look again — With one eliminated and another analyzed, revisit your original pick. Ask: *Does it build naturally and causally* from the scene? Is it *visually, emotionally, and logically fitting?* *Re-affirm its strength by explicitly linking it to the scene's prior state or action.*
O: Opt with confidence — Justify your final answer. Explain why it's the most coherent, contextually appropriate, and direct continuation, avoiding any logical or tonal leaps present in other options.

Output the number corresponding to the final answer in the format:
ANSWER: number

Example 1:
Question: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then
Options:
0: the man adds wax to the windshield and cuts it.
1: a person boards a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.
2: the man puts on a Christmas coat, knitted with netting.
3: the man continues removing the snow on his car.


P: I choose option 3 because it naturally continues the snowy car scenario. The action of "removing the snow" is a direct and logical progression from "writes over the snow covering the window," maintaining continuity of action and setting.
O: Option 2 doesn't work — the Christmas coat is unrelated, introduces an unnecessary costume change, and breaks the established practical tone of the scene.
E: Option 0 starts well with “adds wax,” which is related to car maintenance, but “cuts it” is jarring, implausible in this context, and inconsistent with clearing snow. It introduces a contradictory action.
L: Option 3 still holds. It directly follows the action of clearing snow from a car, making it a causally and visually fitting next step. The action is consistent with the scene's focus.
O: Option 3 is realistic and coherent. It preserves continuity by presenting the most direct and logical continuation of the man's activity with the snow-covered car, avoiding any extraneous or inconsistent elements.
ANSWER: 3


Example 2:
Question: He takes his lighter and lights the newspaper in several places to start the fire. The bonfire starts burning and continues to burn. he
Options:
0: plays with the dog and makes two cookies.
1: adds a few more twigs to keep the flames burning.
2: gets up and attempts to put a flag on it, fails and makes a complete ass out of himself.
3: puts on equipment and stools.


P: I choose option 1 because it logically continues the act of managing the bonfire. Adding twigs is a direct, practical, and immediate next action to sustain a burning fire.
O: Option 3 makes little sense — “puts on equipment and stools” is vague, doesn't tie to the fire-starting context, and introduces unrelated objects, breaking continuity of purpose.
E: Option 2 is overly theatrical, introduces a completely new, unrelated goal (putting on a flag), and doesn’t fit the purposeful tone of fire tending. It's an implausible immediate follow-up.
L: Option 1 still makes sense. Adding twigs aligns directly with maintaining a fire, both causally and visually, making it the most immediate and logical continuation of the burning bonfire.
O: Option 1 fits well — it's purposeful, sequentially sound, and the most direct and realistic action to ensure the fire continues, avoiding any tonal or logical jumps present in the other options.
ANSWER: 1


Example 3:
Question: The roof is done and a view of the entire house is shown to show off the finished roof. the woman
Options:
0: is standing in front of the home, smiling while talking.
1: interviews the man again and leaves the room.
2: shows the soil with two lay-ups of shingle and applies a layer onto the top shingle.
3: stacks the bags on the side and begins putting stencils on the top.


P: I choose option 0 because it reflects a natural and common celebratory or appreciative moment after completing a significant project like a roof. It maintains the visual focus on the finished house.
O: Option 1 takes us back indoors, introduces an unrelated "interview," and breaks the scene's continuity, which is focused on the *finished* roof and house exterior.
E: Option 2 talks about applying shingles again, which directly contradicts the earlier claim that the roof is already "done." This introduces a significant logical inconsistency.
L: Option 0 remains consistent with the visual conclusion of the project and the implied emotional tone of satisfaction. It's a natural social interaction in response to a completed task.
O: Option 0 logically follows a finished roof scene. Standing, smiling, and talking in front of the completed home fits both the action (showing off the roof) and the celebratory tone, making it the most coherent and contextually appropriate choice.
ANSWER: 0
'''

template_3 = '''
You are a thoughtful reasoner using the METAL framework:


M: Make a claim — state which answer you think is correct. 
E: Provide evidence — explain why that answer fits best. 
T: Think of a counter — consider another choice that might seem plausible. 
A: Acknowledge limitations — note any uncertainty or assumptions. 
L: Learn from it — reflect on what makes the chosen answer better overall.

Given a multiple-choice question (MCQ), select the number (0, 1, 2, or 3) corresponding to the correct answer. 
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
A: While we don't know his full intent, the narrative tone seems functional rather than chaotic or comedic. 
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
T: Option 2 might be tempting if the roof weren't already declared “done,” suggesting more work. 
A: It's possible the video could jump back in time, but that's unlikely given the “roof is done” framing. 
L: Option 0 matches the narrative closure style often used in documentary or instructional formats. 
ANSWER: 0
'''

# STEP 1: Construct batch requests
batch_requests = []
# for original_idx, train_data_point in zip(all_overlapping_elements, selected_data):
for original_idx, train_data_point in zip(random_indices, selected_data):
    #prompt = "Select number corresponding to the correct answer\n"
    prompt = f"Question: {train_data_point['ctx']}\n"
    for opt_idx, option in enumerate(train_data_point['endings']):
        prompt += f"{opt_idx}: {option} \n"

    prompt_additions = []
    prompt_additions.append(prompt)
    prompt_additions.append("### Agent Responses: \n\n")
    prompt_additions.append("## Prompting Strategy 1: ")
    prompt_additions.append(template_1)
    prompt_additions.append("Prompting Strategy 1 Response: ")
    prompt_additions.append(df.loc[df['original_idx'] == original_idx, 'response_content_1'].values[0])

    prompt_additions.append("## Prompting Strategy 2: ")
    prompt_additions.append(template_2)
    prompt_additions.append("Prompting Strategy 2 Response: ")
    prompt_additions.append(df.loc[df['original_idx'] == original_idx, 'response_content_2'].values[0])

    prompt_additions.append("## Prompting Strategy 3: ")
    prompt_additions.append(template_3)
    prompt_additions.append("Prompting Strategy 3 Response: ")
    prompt_additions.append(df.loc[df['original_idx'] == original_idx, 'response_content_3'].values[0])

    prompt = "\n\n".join(prompt_additions)

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

cur_input_file = "batch_input_gpt_3.5_turbo_judge_5.jsonl"

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

