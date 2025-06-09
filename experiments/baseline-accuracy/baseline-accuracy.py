# Baseline Accuracy Experiment
# This experiment attempts to benchmark the accuracy of the baseline model on QA1-QA5. 
# To run it, ensure that the `allow_prompt` variable is set to `True`. 
# To remove all preexisting results (if any) set `reset_experiment` to `True`.

import glob
import os
from pathlib import Path
import sys
import time

from openai import OpenAI
from openai import InternalServerError as OpenAIInternalServerError

import pandas as pd

from ReES.babilong import BABILongTargetBenchmarkAdapter, BABILongQuestionType
from ReES.env import OPENAI_API_KEY

from tqdm.auto import tqdm

sys.path.append("..")

folder_path = Path(__file__).parent
dataset_path = Path(folder_path / "../.." / "target_dataset")

adapter = BABILongTargetBenchmarkAdapter(dataset_path=dataset_path)

allow_prompt = True

reset_experiment = False
if reset_experiment:
    # Delete all .csv and .json files in current directory
    for file_path in glob.glob("*.csv") + glob.glob("*.json"):
        os.remove(file_path)
        print(f"Deleted: {file_path}")

client = OpenAI(
  api_key=f"{OPENAI_API_KEY}",
  base_url="https://api.openai.com/v1/"
)
model_name = "gpt-4.1-nano-2025-04-14"

output_filename = "baseline_accuracy"

# completion = client.chat.completions.create(
#   model = model_name,
#   store = True,
#   messages=[
#     {"role": "system", "content": "You are a intelligent assistant."},
#     {"role": "user", "content": "write a haiku about compilers"}
#   ]
# )
# print(completion)
# print(completion.choices[0].message)

## All credit goes to BABILong for this cell. The original code can be found at https://github.com/booydar/babilong/blob/main/babilong/prompts.py
## The prompt templates are used to reproduce BABILong's original baseline experiment, but with a new baseline model (ChatGPT 4.1 nano)
## And with a different datasetup (local, not Hugging Face) and output (for our own further processing)
TASK_TEMPLATE = '{instruction}\n\n{examples}\n\n{post_prompt}'
USER_TEMPLATE = '<context>\n{context}\n</context>\n\nQuestion: {question}'
DEFAULT_TEMPLATE = f'{TASK_TEMPLATE}\n\n{USER_TEMPLATE}'

def get_formatted_input(context, question, examples, instruction, post_prompt, template=DEFAULT_TEMPLATE):
    # instruction - task instruction
    # examples - in-context examples
    # post_prompt - any additional instructions after examples
    # context - text to use for qa
    # question - question to answer based on context
    formatted_input = template.format(instruction=instruction, examples=examples, post_prompt=post_prompt,
                                      context=context.strip(), question=question)
    return formatted_input.strip()

# Only default prompts for qa1-qa5 are copied
DEFAULT_PROMPTS = {
    'qa1': {
        'instruction':
            'I will give you context with the facts about positions of different persons hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts. '
            'If a person was in different locations, use the latest location to answer the question.',
        'examples':
            '<example>\n'
            'Charlie went to the hallway. Judith come back to the kitchen. Charlie travelled to balcony. '
            'Where is Charlie?\n'
            'Answer: The most recent location of Charlie is balcony.\n'
            '</example>\n\n'
            '<example>\n'
            'Alan moved to the garage. Charlie went to the beach. Alan went to the shop. Rouse '
            'travelled to balcony. Where is Alan?\n'
            'Answer: The most recent location of Alan is shop.\n'
            '</example>',
        'post_prompt':
            'Always return your answer in the following format: '
            'The most recent location of ’person’ is ’location’. Do not write anything else after that.'
    },
    'qa2': {
        'instruction':
            'I give you context with the facts about locations and actions of different persons '
            'hidden in some random text and a question.'
            'You need to answer the question based only on the information from the facts.\n'
            'If a person got an item in the first location and travelled to the second location '
            'the item is also in the second location. '
            'If a person dropped an item in the first location and moved to the second location '
            'the item remains in the first location.',
        'examples':
            '<example>\n'
            'Charlie went to the kitchen. Charlie got a bottle. Charlie moved to the balcony. '
            'Where is the bottle?\n'
            'Answer: The bottle is in the balcony.\n'
            '</example>\n'
            '<example>\n'
            'Alan moved to the garage. Alan got a screw driver. Alan moved to the kitchen. Where '
            'is the screw driver?\n'
            'Answer: The screw driver is in the kitchen.\n'
            '</example>',
        'post_prompt':
            'Always return your answer in the following format: The ’item’ is in ’location’. '
            'Do not write anything else after that.'
    },
    'qa3': {
        'instruction':
            'I give you context with the facts about locations and actions of different persons '
            'hidden in some random text and a question. '
            'You need to answer the question based only on the information from the facts.\n'
            'If a person got an item in the first location and travelled to the second location '
            'the item is also in the second location. '
            'If a person dropped an item in the first location and moved to the second location '
            'the item remains in the first location.',
        'examples':
            '<example>\n'
            'John journeyed to the bedroom. Mary grabbed the apple. Mary went back to the bathroom. '
            'Daniel journeyed to the bedroom. Daniel moved to the garden. Mary travelled to the kitchen. '
            'Where was the apple before the kitchen?\n'
            'Answer: Before the kitchen the apple was in the bathroom.\n'
            '</example>\n'
            '<example>\n'
            'John went back to the bedroom. John went back to the garden. John went back to the kitchen. '
            'Sandra took the football. Sandra travelled to the garden. Sandra journeyed to the bedroom. '
            'Where was the football before the bedroom?\n'
            'Answer: Before the bedroom the football was in the garden.\n'
            '</example>',
        'post_prompt':
            'Always return your answer in the following format: '
            'Before the $location_1$ the $item$ was in the $location_2$. Do not write anything else after that.'
    },
    'qa4': {
        'instruction':
            'I will give you context with the facts about different people, their location and actions, hidden in '
            'some random text and a question. '
            'You need to answer the question based only on the information from the facts.',
        'examples':
            '<example>\n'
            'The hallway is south of the kitchen. The bedroom is north of the kitchen. '
            'What is the kitchen south of?\n'
            'Answer: bedroom\n'
            '</example>\n'
            '<example>\n'
            'The garden is west of the bedroom. The bedroom is west of the kitchen. What is west of the bedroom?\n'
            'Answer: garden\n'
            '</example>',
        'post_prompt':
            'Your answer should contain only one word - location. Do not write anything else after that.'
    },
    'qa5': {
        'instruction':
            'I will give you context with the facts about locations and their relations hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts.',
        'examples':
            '<example>\n'
            'Mary picked up the apple there. Mary gave the apple to Fred. Mary moved to the bedroom. '
            'Bill took the milk there. Who did Mary give the apple to?\n'
            'Answer: Fred\n'
            '</example>\n'
            '<example>\n'
            'Jeff took the football there. Jeff passed the football to Fred. Jeff got the milk there. '
            'Bill travelled to the bedroom. Who gave the football?\n'
            'Answer: Jeff\n'
            '</example>\n'
            '<example>\n'
            'Fred picked up the apple there. Fred handed the apple to Bill. Bill journeyed to the bedroom. '
            'Jeff went back to the garden. What did Fred give to Bill?\n'
            'Answer: apple\n'
            '</example>',
        'post_prompt':
            'Your answer should contain only one word. Do not write anything else after that. '
            'Do not explain your answer.'
    }
}

use_instruction = True
use_examples = True
use_post_prompt = True
use_chat_template = True

## The following code belongs to BABILong and can be found at https://github.com/booydar/babilong/blob/main/notebooks/babilong_eval_openai_api_models.ipynb
## It is copied to run our experiment in a setting as close to the original BABILong experiment as possible
 
# define generation parameters
generate_kwargs = {
    'n': 1,
    'temperature': 0.0,
}

def compare_answers(target, output):
    return target.lower() == output.strip().lower().strip('".')


text_token_sizes = [0, 1, 2, 4, 8, 16, 32, 64, 128]
qa_types = [BABILongQuestionType.qa1, BABILongQuestionType.qa2, BABILongQuestionType.qa3, BABILongQuestionType.qa4, BABILongQuestionType.qa5]
first_text_id = 0
text_count = 50
text_ids = [x for x in range(first_text_id, first_text_id + text_count)] # We might not want to run all texts
text_ids

TEMPLATE = '{instruction}\n{examples}\n{post_prompt}\nContext: {context}\n\nQuestion: {question}'

# prompt_cfg for few-shot
"""
prompt_cfg = {
    'instruction': DEFAULT_PROMPTS[qa.value]['instruction'] if use_instruction else '',
    'examples': DEFAULT_PROMPTS[qa.value]['examples'] if use_examples else '',
    'post_prompt': DEFAULT_PROMPTS[qa.value]['post_prompt'] if use_post_prompt else '',
    'template': DEFAULT_TEMPLATE,
    'chat_template': use_chat_template,
}
"""
# for zero-shot
prompt_cfg = {
        'instruction': 'Answer only with 1 word. \n',
        'examples': '', 
        'post_prompt': '',
        'template': TEMPLATE,
}

# Loop over the different question types

columns = [
    "qa", "token_size", "text_id", "question", "target", "llm_output", "correct_guess", "input_tokens", "output_tokens", "total_token_usage", "time_used_llm", "time_used_total"
]
csv_file = f"{output_filename}.csv"
json_file = f"{output_filename}.json"

if os.path.exists(csv_file):
    results = pd.read_csv(csv_file)
    processed_ids = set(zip(results['qa'], results['token_size'], results['text_id']))
    print(f"Resuming from existing file with {len(results)} entries.")
else:
    results = pd.DataFrame(columns=columns)
    processed_ids = set()


for qa in tqdm(qa_types, desc="qa-types"):
    # Configure BABILong's prompt setup, see references above for the source.
    prompt_name = [f'{k}_yes' if prompt_cfg[k] else f'{k}_no' for k in prompt_cfg if k != 'template']
    prompt_name = '_'.join(prompt_name)

    

    # Loop over the different token sizes
    for token_size in tqdm(text_token_sizes, desc="token-sizes"):
        dataset = adapter.batch_fetch_texts(
            token_size=token_size,
            question_types=[qa],
            text_ids=text_ids
            )[qa]
        
        # Loop over each text
        for i, text in enumerate(tqdm(dataset, desc=f"Texts for {qa}-{token_size}k "), start = first_text_id):
            if (qa.value, token_size, i) in processed_ids:
                continue
            
            text_time_start = time.time()
    
            # Construct input to LLM according to BABILong's own structure
            llm_input_text = get_formatted_input(
                context = text.text,
                question = text.target_question,
                examples = prompt_cfg['examples'],
                instruction = prompt_cfg['instruction'],
                post_prompt = prompt_cfg['post_prompt'],
                template = prompt_cfg['template']
            )

            # Construct the messages to give to the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are a intelligent assistant."
                },
                {
                    "role": "user",
                    "content": llm_input_text,
                },
            ]
            

            llm_response = None
            llm_output = ""

            time_used_llm = 0
            
            if (allow_prompt):
                try:
                    llm_time_start = time.time()
                    llm_response = client.chat.completions.create(
                        model=model_name, 
                        messages=messages, 
                        n = generate_kwargs['n'],
                        temperature = generate_kwargs['temperature']
                    )
                    llm_time_end = time.time()
                    time_used_llm = llm_time_end - llm_time_start


                    
                    if llm_response is not None and llm_response.choices is not None:
                        llm_output = llm_response.choices[0].message.content.strip()

                    llm_answer_correct = compare_answers(
                        target = text.target,
                        output = llm_output,
                    )

                    text_time_end = time.time()                
                    time_used_total = text_time_end - text_time_start

                    text_results = {
                        "qa": qa.value,
                        "token_size": token_size,
                        "text_id": i,
                        "question": text.target_question,
                        "target": text.target,
                        "llm_output": llm_output,
                        "correct_guess": llm_answer_correct,
                        "input_tokens": llm_response.usage.prompt_tokens,
                        "output_tokens": llm_response.usage.completion_tokens,
                        "total_token_usage": llm_response.usage.total_tokens,
                        "time_used_llm": time_used_llm,
                        "time_used_total": time_used_total,
                    }

                    results.loc[len(results)] = text_results
                    results.to_csv(csv_file, index=False)
                    results.to_json(json_file, orient='records', indent = 4)
                except OpenAIInternalServerError as e:
                    # If an error happens, write it, so we can identify the errored text. But else continue
                    error_string = f"{qa.value}, {token_size}, {i}\n"
                    print(e, error_string)
                    with open('baseline_accuracy_errors', 'a') as error_file:                        
                        error_file.write(error_string)
                    error_file.close()
            
            else:
                print("LLM Prompt not allowed by own settings")
            time.sleep(0.5) # Ensure never hitting api rate limit
            