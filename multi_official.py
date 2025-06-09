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
from openai import OpenAI
import pandas
import pickle
from helper import *

exp_run = "METAL_POELO_MULTI_AGENT"

# Define number of agents in experiment
num_agents = 2 

# Prompt template 
template_1 = '''
You are a thoughtful reasoner using the METAL framework:

M: Make a claim — state which answer you think is correct.
E: Provide evidence — explain why that answer fits best.
T: Think of a counter — consider another choice that might seem plausible.
A: Acknowledge limitations — note any uncertainty or assumptions.
L: Learn from it — reflect on what makes the chosen answer better overall.


Given a multiple-choice question (MCQ), select the number (0, 1, 2, or 3) corresponding to the correct answer.
Do not output the answer until all 5 steps are completed.
End with: ANSWER: <number>

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


template_2 = '''
You will get output from another reasoning agent. Evaluate the agents output by
applying the POELO method.


P: Preview and pick — Read the four options carefully. Which one seems like the most *realistic and natural* continuation of the situation? Pick that first, *explicitly stating why it maintains continuity of action, setting, or character intent*.
O: Outrule one — Identify and eliminate one option that clearly *breaks continuity* (e.g., jumps to a new, unrelated scene), *introduces something implausible or contradictory*, or *drastically alters the established tone*. Explain *why* it fails based on these criteria.
E: Examine another — Look at a tempting but imperfect option. Why is it *less appropriate* than your first choice? Focus on how it *disrupts logical flow, introduces minor inconsistencies, or is less direct/efficient* in continuing the scene.
L: Look again — With one eliminated and another analyzed, revisit your original pick. Ask: *Does it build naturally and causally* from the scene? Is it *visually, emotionally, and logically fitting?* *Re-affirm its strength by explicitly linking it to the scene's prior state or action.*
O: Opt with confidence — Justify your final answer. Explain why it’s the most coherent, contextually appropriate, and direct continuation, avoiding any logical or tonal leaps present in other options.


Only after completing all 5 steps, end with: ANSWER: <number>


Example 1:
Question: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then
Options:
0: the man adds wax to the windshield and cuts it.
1: a person boards a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.
2: the man puts on a Christmas coat, knitted with netting.
3: the man continues removing the snow on his car.

P: I choose option 3 because it naturally continues the snowy car scenario. The action of "removing the snow" is a direct and logical progression from "writes over the snow covering the window," maintaining continuity of action and setting.
O: Option 2 doesn’t work — the Christmas coat is unrelated, introduces an unnecessary costume change, and breaks the established practical tone of the scene.
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
O: Option 3 makes little sense — “puts on equipment and stools” is vague, doesn’t tie to the fire-starting context, and introduces unrelated objects, breaking continuity of purpose.
E: Option 2 is overly theatrical, introduces a completely new, unrelated goal (putting on a flag), and doesn’t fit the purposeful tone of fire tending. It's an implausible immediate follow-up.
L: Option 1 still makes sense. Adding twigs aligns directly with maintaining a fire, both causally and visually, making it the most immediate and logical continuation of the burning bonfire.
O: Option 1 fits well — it’s purposeful, sequentially sound, and the most direct and realistic action to ensure the fire continues, avoiding any tonal or logical jumps present in the other options.
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


Final answer: ANSWER: <number>


YOU MUST ANSWER AT THE END WITH: ANSWER: <number>
'''

api_key = #provide api key

ds = load_dataset("Rowan/hellaswag")
client = openai.OpenAI(api_key=api_key)

templates = []
for i in range(1, num_agents + 1): 
    templates.append(eval(f"template_{i}"))

random.seed(42)
random_indices = random.sample(range(len(ds['train'])), 400)
selected_data = ds['train'].select(random_indices)

# MAIN 
outputs_into_inputs = None 
max_retries = 15

for i in range(num_agents):
    template = templates[i]
    inputs = create_input(outputs_into_inputs, template, random_indices, selected_data)

    cur_input_file = f"batch_input_agent_{i+1}.jsonl"
    with open(cur_input_file, "w") as f:
        for request in inputs:
            f.write(json.dumps(request) + "\n")


    for attempt in range(0, max_retries + 1):
        with open(cur_input_file, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")

        response = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        batch_id = response.id
        print(f"Submitted batch for Agent {i+1}: {batch_id}")

        while True:
            status = client.batches.retrieve(batch_id).status
            print(f"Batch status: {status}")
            if status in ["completed", "failed", "expired"]:
                break
            time.sleep(20)

        final_status = client.batches.retrieve(batch_id).status
        if final_status == "completed":
            print(f"Batch for Agent {i+1} completed on attempt {attempt}")
            break
        else:
            print(f"Batch for Agent {i+1} failed on attempt {attempt}")
            continue 

    if i != num_agents - 1:
        outputs_into_inputs = extract_outputs(batch_id, client)
        
        # Save intermediate outputs to JSON
        intermediate_json_path = f"{exp_run}_outputs_into_inputs_agent_{i+1}.json"
        with open(intermediate_json_path, "w") as f:
            json.dump(outputs_into_inputs, f, indent=2)

    else:
        print("ENTERED HERE")
        final_answers = get_answer(batch_id, client, ds)

        # Save final output to JSON
        with open(f"{exp_run}_final_answers.json", "w") as f:
            json.dump(final_answers, f, indent=2)