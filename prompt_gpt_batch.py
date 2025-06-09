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


def extract_label(output_text):
    match = re.search(r'ANSWER:\s*(\d+)', output_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None

our_key = # add api key please

ds = load_dataset("Rowan/hellaswag")

# i = 0
# # Loop through the first 2 data points
# for i, data_point in enumerate(ds["train"]):
#     prompt = f"Question: {data_point['ctx']}\n"
#     for opt_idx, option in enumerate(data_point['endings']):
#         prompt += f"{opt_idx}: {option} \n"

#     print(i)
#     print(prompt)
#     print(data_point['label'])

#     if i == 50:  # Stop after 2 examples (i=0 and i=1)
#         break


# exit()
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

# Make sure to carefully reason and then answer. 
# Before outputting ensure the final answer format has the word ANSWER and a 
# label 0, 1, 2, 3. 
# '''

# template = '''
# You are an expert problem solver. Read the MCQ and before answering, briefly go through:

# 1. Identify: What’s happening?
# 2. Define: What’s the core question?
# 3. Explore: List 2–3 plausible endings.
# 4. Act: Pick the best ending.
# 5. Look back: In one sentence, why it’s best.

# Important: At the end, output your answer in this exact format (with nothing before or after it):
# ANSWER: 1 (or 0, 2, or 3 depending on your answer — just the number, no brackets or repetition)

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

# template = '''
# You are an expert problem solver using the IDEAL framework:

# I: Identify the situation.  
# D: Define the problem — what needs to be figured out or predicted.  
# E: Explore possible options and evaluate them.  
# A: Act by selecting the best choice.  
# L: Look back and justify your decision.

# Given a multiple-choice question (MCQ), select the number (0, 1, 2, or 3) corresponding to the correct answer.  
# Do not output the answer until all 5 steps are completed.  
# End with: ANSWER: number

# ---

# Example 1  
# Question:  
# People ride horses in the shallow waters of the sea. People ride the horses in the water along the shore. A horse defecates in the water.  
# people  
# 0: form a line with the horses while riding in the ocean.  
# 1: swim in the choppy water.  
# 2: stand on the horses giving the people water water when children stand on the sides.  
# 3: assist the horses in the water.

# I: People are riding horses in shallow water, and a horse defecates.  
# D: We need to infer what people plausibly do next.  
# E:  
# - 0: Plausible activity, but doesn’t respond to the defecation.  
# - 1: Abrupt change in focus.  
# - 2: Gibberish.  
# - 3: Makes sense — people may help the horse or react.  
# A: Option 3 responds to the defecation event and is coherent.  
# L: It continues the situation sensibly and plausibly.  
# ANSWER: 3

# ---

# Example 2  
# Question:  
# Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles.  
# then  
# 0: , the man adds wax to the windshield and cuts it.  
# 1: , a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.  
# 2: , the man puts on a christmas coat, knitted with netting.  
# 3: , the man continues removing the snow on his car.

# I: A man is writing on snowy glass; a woman smiles.  
# D: What is the logical next step in the snowy car context?  
# E:  
# - 0: Doesn’t make sense contextually — waxing and cutting is unrelated.  
# - 1: Incoherent grammar.  
# - 2: Random outfit change.  
# - 3: Logical — he keeps working on snow removal.  
# A: Option 3 maintains the narrative and setting.  
# L: Most coherent and task-aligned continuation.  
# ANSWER: 3

# ---

# Example 3  
# Question:  
# A female chef in white uniform shows a stack of baking pans in a large kitchen presenting them. the pans  
# 0: contain egg yolks and baking soda.  
# 1: are then sprinkled with brown sugar.  
# 2: are placed in a strainer on the counter.  
# 3: are filled with pastries and loaded into the oven.

# I: A chef is presenting baking pans in a kitchen.  
# D: What happens next to the baking pans?  
# E:  
# - 0: Descriptive but not an action.  
# - 1: Unclear and isolated.  
# - 2: Illogical use of a strainer.  
# - 3: Logical next step — the pans are used for baking.  
# A: Choose 3 — the narrative flows best.  
# L: It completes the implied baking activity naturally.  
# ANSWER: 3

# ---

# Now solve the next question:

# Question:  
# <Insert your new MCQ here> '''

# template = '''
# You are an expert problem solver using the IDEAL framework:

# I: Identify the situation.  
# D: Define the problem — what needs to be figured out or predicted.  
# E: Explore possible options and evaluate them.  
# A: Act by selecting the best choice.  
# L: Look back and justify your decision.

# Given a multiple-choice question (MCQ), select the number (0, 1, 2, or 3) corresponding to the correct answer.  
# Do not output the answer until all 5 steps are completed.  
# End with: ANSWER: number

# Example 1:
# Question: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then  
# Options:  
# 0: the man adds wax to the windshield and cuts it.  
# 1: a person boards a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.  
# 2: the man puts on a Christmas coat, knitted with netting.  
# 3: the man continues removing the snow on his car.

# I: The scene involves a man interacting with snow on a car and a smiling woman nearby — likely a winter setting with mundane or cheerful tone.  
# D: We need to predict the man's next action, staying consistent with the calm, realistic setting.  
# E:  
# - Option 0 involves dangerous or implausible behavior.  
# - Option 1 is incoherent and introduces many unrelated characters and actions.  
# - Option 2 involves a costume change without context.  
# - Option 3 logically follows — the man keeps clearing snow.  
# A: Choose 3, which is natural and continues the described activity.  
# L: This maintains narrative flow and realistic behavior in a snowy environment.  
# ANSWER: 3

# Example 2:
# Question: He takes his lighter and lights the newspaper in several places to start the fire. The bonfire starts burning and continues to burn. he  
# Options:  
# 0: plays with the dog and makes two cookies.  
# 1: adds a few more twigs to keep the flames burning.  
# 2: gets up and attempts to put a flag on it, fails and makes a complete ass out of himself.  
# 3: puts on equipment and stools.

# I: A fire has just been lit, and it continues to burn — this is a fire-tending scenario.  
# D: Predict what logically follows when someone wants to keep a fire going.  
# E:  
# - Option 0 is unrelated to fire maintenance.  
# - Option 1 fits naturally as a next step to sustain the fire.  
# - Option 2 is humorous but irrelevant and inconsistent.  
# - Option 3 is vague and disconnected.  
# A: Choose 1, which logically follows lighting a fire.  
# L: Adding twigs to maintain a flame is standard fire-tending behavior.  
# ANSWER: 1

# Example 3:
# Question: The roof is done and a view of the entire house is shown to show off the finished roof. the woman  
# Options:  
# 0: is standing in front of the home, smiling while talking.  
# 1: interviews the man again and leaves the room.  
# 2: shows the soil with two lay-ups of shingle and applies a layer onto the top shingle.  
# 3: stacks the bags on the side and begins putting stencils on the top.

# I: The house roof is completed, and the scene shifts to a wide shot — suggesting a wrap-up or presentation moment.  
# D: Predict what the woman does during this likely concluding or showcasing scene.  
# E:  
# - Option 0 shows her standing and talking, a common wrap-up visual.  
# - Option 1 contradicts the outdoor setting and context.  
# - Option 2 implies roofing work is still happening — but the roof is already done.  
# - Option 3 introduces irrelevant actions (stencils?) not tied to roofing.  
# A: Select 0, as it aligns with the narrative of showing off the finished work.  
# L: This action is consistent with documentary or presentation-style footage.  
# ANSWER: 0
# '''

# METAL 
# template = '''
# You are a thoughtful reasoner using the METAL framework:

# M: Make a claim — state which answer you think is correct.  
# E: Provide evidence — explain why that answer fits best.  
# T: Think of a counter — consider another choice that might seem plausible.  
# A: Acknowledge limitations — note any uncertainty or assumptions.  
# L: Learn from it — reflect on what makes the chosen answer better overall.

# Given a multiple-choice question (MCQ), select the number (0, 1, 2, or 3) corresponding to the correct answer.  
# Do not output the answer until all 5 steps are completed.  
# End with: ANSWER: number

# Example 1:
# Question: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then  
# Options:  
# 0: the man adds wax to the windshield and cuts it.  
# 1: a person boards a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.  
# 2: the man puts on a Christmas coat, knitted with netting.  
# 3: the man continues removing the snow on his car.

# M: I claim that option 3 is correct.  
# E: The man is already interacting with snow on a car, so continuing to remove it is a logical next step.  
# T: Option 2 might seem plausible since it's winter and he might put on winter gear.  
# A: However, there's no prior mention of needing new clothes or a costume change — and option 2 lacks relevance.  
# L: The clearest progression is from writing in snow to removing it, making 3 the most natural continuation.  
# ANSWER: 3

# Example 2:
# Question: He takes his lighter and lights the newspaper in several places to start the fire. The bonfire starts burning and continues to burn. he  
# Options:  
# 0: plays with the dog and makes two cookies.  
# 1: adds a few more twigs to keep the flames burning.  
# 2: gets up and attempts to put a flag on it, fails and makes a complete ass out of himself.  
# 3: puts on equipment and stools.

# M: I claim that option 1 is correct.  
# E: Adding twigs is a reasonable action when tending a new fire to sustain it.  
# T: Option 2 might seem like it adds drama or humor, but it veers into implausibility.  
# A: While we don’t know his full intent, the narrative tone seems functional rather than chaotic or comedic.  
# L: Option 1 stays focused on maintaining the fire, which is the core of the scene.  
# ANSWER: 1

# Example 3:
# Question: The roof is done and a view of the entire house is shown to show off the finished roof. the woman  
# Options:  
# 0: is standing in front of the home, smiling while talking.  
# 1: interviews the man again and leaves the room.  
# 2: shows the soil with two lay-ups of shingle and applies a layer onto the top shingle.  
# 3: stacks the bags on the side and begins putting stencils on the top.

# M: I claim that option 0 is correct.  
# E: Showing off a finished roof followed by the woman smiling in front of the house aligns with a project wrap-up moment.  
# T: Option 2 might be tempting if the roof weren’t already declared “done,” suggesting more work.  
# A: It’s possible the video could jump back in time, but that’s unlikely given the “roof is done” framing.  
# L: Option 0 matches the narrative closure style often used in documentary or instructional formats.  
# ANSWER: 0
# '''

#SELFIE 
# template = '''
# You are a reflective and self-aware problem solver using the SELFIE strategy:

# S: State the answer — what you believe is the best option.  
# E: Explain your reasoning clearly.  
# L: Consider limitations in your reasoning or the context.  
# F: Try to falsify your own claim — challenge your answer and consider a plausible alternative.  
# I: Interrogate any ambiguity in the question or answer choices.  
# E: End with: ANSWER: number

# Use SELFIE to answer each multiple-choice question (MCQ).  
# Choose from options 0, 1, 2, or 3.

# Example 1:
# Question: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then  
# Options:  
# 0: the man adds wax to the windshield and cuts it.  
# 1: a person boards a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.  
# 2: the man puts on a Christmas coat, knitted with netting.  
# 3: the man continues removing the snow on his car.

# S: I choose option 3.  
# E: The man is already interacting with the snow and the car, so continuing to remove it makes narrative sense.  
# L: I’m assuming this is a realistic, sequential scene — if it were symbolic or comedic, I might miss the joke.  
# F: Option 2 could be plausible in a holiday montage, but there’s no hint of that framing here.  
# I: Option 1 is grammatically broken and hard to parse. Option 0 introduces “cutting” a windshield, which feels violent and out of place.  
# ANSWER: 3

# Example 2:
# Question: He takes his lighter and lights the newspaper in several places to start the fire. The bonfire starts burning and continues to burn. he  
# Options:  
# 0: plays with the dog and makes two cookies.  
# 1: adds a few more twigs to keep the flames burning.  
# 2: gets up and attempts to put a flag on it, fails and makes a complete ass out of himself.  
# 3: puts on equipment and stools.

# S: I choose option 1.  
# E: The scene is about fire-starting. Adding twigs is the clearest continuation of that activity.  
# L: I’m assuming the tone is practical, not humorous or chaotic.  
# F: If the scene were comedic or absurd, option 2 might match — but it breaks the tone.  
# I: Option 3 is vague. “Puts on equipment and stools” isn’t a coherent or related action.  
# ANSWER: 1

# Example 3:
# Question: The roof is done and a view of the entire house is shown to show off the finished roof. the woman  
# Options:  
# 0: is standing in front of the home, smiling while talking.  
# 1: interviews the man again and leaves the room.  
# 2: shows the soil with two lay-ups of shingle and applies a layer onto the top shingle.  
# 3: stacks the bags on the side and begins putting stencils on the top.

# S: I choose option 0.  
# E: Showing off the finished house aligns with someone proudly standing in front of it and smiling.  
# L: I assume the narrative is linear — if it’s nonlinear or symbolic, another choice could be valid.  
# F: Option 2 could make sense if we were still mid-project, but the roof is already finished.  
# I: Option 3 is confusing — “putting stencils on the top” is vague and doesn’t clearly relate to roofing.  
# ANSWER: 0
# '''

# HYBRID 
template = '''
You are a rigorous, reflective reasoner who uses a hybrid of three expert thinking strategies:  
SELFIE, IDEAL, and METAL. Use them together to answer multiple-choice questions (MCQs).

Your process:

1. **State your claim** — what is the best answer and why.  
2. **Explain your reasoning** — show how the facts or logic support your choice.  
3. **Define the problem** — what must be predicted, completed, or inferred.  
4. **Explore alternatives and try to falsify your claim** — consider other options seriously.  
5. **Consider limitations and ambiguity** — what could make your choice uncertain?  
6. **Learn from your decision** — what general principle or insight did you apply?  
7. End with: `ANSWER: number`

Use this 7-step structure for each example below:

---

**Example 1**  
Question: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then  
Options:  
0: the man adds wax to the windshield and cuts it.  
1: a person boards a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.  
2: the man puts on a Christmas coat, knitted with netting.  
3: the man continues removing the snow on his car.

1. **State your claim**: I believe option 3 is the correct answer.  
2. **Explain reasoning**: The scene is grounded in wintertime and car-related activity. If he was writing in snow, continuing to remove it is a logical, physical progression.  
3. **Define the problem**: We are predicting the next step in a mundane, coherent winter narrative.  
4. **Explore/falsify**: Option 2 might seem thematically consistent with “winter,” but there's no hint of costume changes. Option 1 is grammatically broken and nonsensical.  
5. **Limitations/ambiguity**: There is some ambiguity in what “writes over snow” means — playful vs. practical.  
6. **Learn**: A grounded physical setting often follows through with realistic next steps unless framed otherwise.  
ANSWER: 3

---

**Example 2**  
Question: He takes his lighter and lights the newspaper in several places to start the fire. The bonfire starts burning and continues to burn. he  
Options:  
0: plays with the dog and makes two cookies.  
1: adds a few more twigs to keep the flames burning.  
2: gets up and attempts to put a flag on it, fails and makes a complete ass out of himself.  
3: puts on equipment and stools.

1. **State your claim**: Option 1 is the best continuation.  
2. **Explain reasoning**: The most direct and logical next step after lighting a fire is to sustain it — adding twigs is typical.  
3. **Define the problem**: What does the man logically do next in a fire-starting sequence?  
4. **Explore/falsify**: Option 2 has narrative flair but contradicts the sober, causal tone. Option 3 is syntactically confusing.  
5. **Limitations/ambiguity**: The question doesn’t specify if this is instructional, comedic, or fictional — tone matters.  
6. **Learn**: When sequences are literal and functional (like fire-building), the next step is usually maintenance, not chaos.  
ANSWER: 1

---

**Example 3**  
Question: The roof is done and a view of the entire house is shown to show off the finished roof. the woman  
Options:  
0: is standing in front of the home, smiling while talking.  
1: interviews the man again and leaves the room.  
2: shows the soil with two lay-ups of shingle and applies a layer onto the top shingle.  
3: stacks the bags on the side and begins putting stencils on the top.

1. **State your claim**: I choose option 0.  
2. **Explain reasoning**: A finished roof followed by a person smiling in front of the house fits a closure or reveal structure.  
3. **Define the problem**: What logically follows a project completion and visual reveal?  
4. **Explore/falsify**: Option 2 implies ongoing work, which contradicts “roof is done.” Option 1 introduces an irrelevant location (“room”).  
5. **Limitations/ambiguity**: The framing could jump in time — back to process steps — but there’s no hint of that.  
6. **Learn**: In sequential documentation or celebration moments, smiling and standing in front of the work is a common conclusion.  
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
random_indices = random.sample(range(len(ds['train'])), 400)
selected_data = ds['train'].select(random_indices)

# STEP 1: Construct batch requests
batch_requests = []
for original_idx, train_data_point in zip(random_indices, selected_data):
    #prompt = "Select number corresponding to the correct answer\n"
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

import json

cur_input_file = "batch_input_IDEAL_best.jsonl"

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

# STEP 4: Download results
# result = client.batches.get_output(batch_id)

# missclassified_original_indices = []
# wrong_answer = []
# none_indices = []
# total_examples = 0

# for item in result:
#     original_idx = int(item["custom_id"])
#     response_content = item["response"]["choices"][0]["message"]["content"]
#     gpt_answer = extract_label(response_content)

#     total_examples += 1
#     if gpt_answer is None:
#         none_indices.append(original_idx)
#         continue

#     gt_label = int(ds['train'][original_idx]['label'])
#     if int(gpt_answer) != gt_label:
#         missclassified_original_indices.append(original_idx)
#         wrong_answer.append(gpt_answer)

# print("Accuracy:", 1 - len(wrong_answer) / total_examples)

# # Save results
# data = (missclassified_original_indices, wrong_answer, none_indices)
# with open('info.pkl', 'wb') as f:
#     pickle.dump(data, f)
