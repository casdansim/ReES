# BABILong's RAG-S and RAG-C accuracy experiment
# This experiments attempts to benchmark the accuracy of the few-shot RAG-S described in the BABILong paper on QA1-QA5.  
# To run it, ensure that the `allow_prompt` variable is set to `True`. 
# To remove all preexisting results (if any) set `reset_experiment` to `True`.

import gc
import glob
import os
from pathlib import Path
import re
import time

from openai import OpenAI
from openai import InternalServerError as OpenAIInternalServerError

import pandas as pd

from ReES.babilong import BABILongQuestionType, BABILongTargetBenchmarkAdapter
from ReES.env import OPENAI_API_KEY

import torch

from tqdm.notebook import tqdm

from transformers import AutoTokenizer, AutoModel


# Load the models
device = "cuda:0"
model_name = "gpt-4.1-nano-2025-04-14"
client = OpenAI(
  api_key=f"{OPENAI_API_KEY}",
  base_url="https://api.openai.com/v1/"
)
output_filename = "RAG"
rag_c_output_file = output_filename + "_C_accuracy"
rag_s_output_file = output_filename + "_S_accuracy"
zero_shot = "zero_shot_"
few_shot = "few_shot_"

# Load the dataset
dataset_path = Path("../../target_dataset/")
adapter = BABILongTargetBenchmarkAdapter(dataset_path=dataset_path)

#Change this value to allow LLM prompts
allow_prompt = True

#Change this value to erase all results (otherwise it will fetch any computed results and resume from that point)
reset_experiment = False
if reset_experiment:
    # Delete all .csv and .json files in current directory
    for file_path in glob.glob("*.csv") + glob.glob("*.json"):
        os.remove(file_path)
        print(f"Deleted: {file_path}")

## All credit goes to BABILong for this cell. The original code can be found at https://github.com/booydar/babilong/blob/main/notebooks/eval_RAG_Llama3.py
## The prompt templates are used to reproduce BABILong's RAG experiment for both the RAG-C and RAG-S, but with a newer model (ChatGPT 4.1 nano)
## And with a different datasetup (local, not Hugging Face) and output (for our own further processing)

## load retriever tokenizer and model
retriever_tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder', device_map=device)
context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder', device_map=device)

generate_kwargs_openai = {
    "temperature": 0.0,
    "top_p": 1.0,
    "stop": None
}

def split_text_to_sent(text):
    # Pattern to split on period, exclamation, or question mark followed by space or end of string
    # Adjust the pattern to handle more edge cases if necessary
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return sentences

def get_formatted_input(messages, context, question, post_prompt):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    formatted_input = system

    if len(messages) > 0:
        if messages[0]['role'] == 'system':
            formatted_input += messages[0]['content'] + '\n\n'
            messages = messages[1:]


        conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) 
        formatted_input += conversation + '\n\n'
    formatted_input += context + "\n\n" + question
    
    if post_prompt: 
        formatted_input += "\n\n" + post_prompt

    formatted_input += " Assistant: "
    return formatted_input

def format_examples(default_examples):
    if len(default_examples) == 0:
        return [], []
    
    examples = default_examples.split('<example>\n')
    examples = [e[:e.index("\n</example>")] for e in examples if len(e) > 0]
    inputs = [e[:e.index("\nAnswer")] for e in examples]
    outputs = [e[e.index("\nAnswer") + 9:] for e in examples]
    return inputs, outputs

def get_messages(context, question, examples, instruction, post_prompt):
    # pre_prompt - general instruction
    # examples - in-context examples
    # post_prompt - any additional instructions after examples
    # context - text to use for qa
    # question - question to answer based on context
    inputs, outputs = format_examples(examples)
    messages = []
    if len(instruction) > 0:
        messages.append({"role": "system", "content": instruction })

    for i, o in zip(inputs, outputs):
        messages += [
            {"role": "user", "content": i},
            {"role": "assistant", "content": o}
        ]

    final_user_message = f"{context}\n\n{question}"
    if post_prompt:
        final_user_message += f"\n\n{post_prompt}"

    messages.append({"role": "user", "content": final_user_message})

    return messages

# Own function using their description of their "division into segments of 512 tokens each" (https://arxiv.org/pdf/2406.10149, appendix G)
def chunk_text_by_tokens(text, tokenizer, chunk_size=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def rag(sample, prompt_cfg, rag_type='C', batchsize=1500):
    assert rag_type in ["C", "S"], "RAG can only be of type 'S' (for sentence length) or 'C' (for chunks of size 512)"

    doc = sample.text
    chunk_list = split_text_to_sent(doc) if rag_type == 'S' else chunk_text_by_tokens(doc, retriever_tokenizer)

    formatted_query_for_retriever = f"User: {sample.target_question}"

    query_input = retriever_tokenizer(formatted_query_for_retriever, return_tensors='pt').to(query_encoder.device)
    with torch.no_grad():
        query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]

    # Initialize lists to store retrieved chunks and their embeddings
    similarities = torch.tensor([]).to(query_emb.device)

    # Process chunks in batches to manage memory usage
    for i in range(0, len(chunk_list), batchsize):  # Adjust the batch size based on your memory constraints
        batch_chunks = chunk_list[i:i+batchsize]
        ctx_input = retriever_tokenizer(batch_chunks, padding=True, truncation=True, max_length=512, return_tensors='pt').to(context_encoder.device)
        with torch.no_grad():
            ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]

        # Compute similarity scores using dot product
        batch_similarities = query_emb.matmul(ctx_emb.transpose(0, 1))  # (1, num_ctx)
        similarities = torch.cat((similarities, batch_similarities), dim=-1)
        
        # Clear memory
        del ctx_input, ctx_emb, batch_chunks
        torch.cuda.empty_cache()
        gc.collect()
    
    ranked_results = torch.argsort(similarities, dim=-1, descending=True)[0][:5]
    retrieved_chunks = [chunk_list[idx] for idx in ranked_results.tolist()]

    del similarities
    torch.cuda.empty_cache()
    gc.collect()

    # Now perform generation with retrieved context
    context = "\n\n".join(retrieved_chunks)

    messages = get_messages(
        context=context, 
        question=sample.target_question,                        
        examples=prompt_cfg['examples'], 
        instruction=prompt_cfg['instruction'],
        post_prompt=prompt_cfg['post_prompt']
    )
    
    llm_time_start = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **generate_kwargs_openai
    )
    llm_time_end = time.time()
    time_used_llm = llm_time_end - llm_time_start

    return response, time_used_llm

def compare_answers(target, output):
    return target.lower() == output.strip().lower().strip('".')

#Defining our setup:
text_token_sizes = [0, 1, 2, 4, 8, 16, 32, 64, 128]
qa_types = [BABILongQuestionType.qa1, BABILongQuestionType.qa2, BABILongQuestionType.qa3, BABILongQuestionType.qa4, BABILongQuestionType.qa5]
first_text_id = 0
text_count = 50
text_ids = [x for x in range(first_text_id, first_text_id + text_count)] # We might not want to run all texts
text_ids

columns = [
    "qa", "token_size", "text_id", "question", "target", "llm_output", "correct_guess", "input_tokens", "output_tokens", "total_token_usage", "time_used_llm", "time_used_total"
]
csv_file = f"{zero_shot+rag_s_output_file}.csv"
json_file = f"{zero_shot+rag_s_output_file}.json"

if os.path.exists(csv_file):
    results = pd.read_csv(csv_file)
    processed_ids = set(zip(results['qa'], results['token_size'], results['text_id']))
    print(f"Resuming from existing file with {len(results)} entries.")
else:
    results = pd.DataFrame(columns=columns)
    processed_ids = set()

# Few shot for RAG-S
TEMPLATE = '{instruction}\n{examples}\n{post_prompt}\nContext: {context}\n\nQuestion: {question}'
rag_type = 'S'

prompt_cfg = {
        'instruction': 'Answer only with 1 word. \n',
        'examples': '', 
        'post_prompt': '',
        'template': TEMPLATE,
}

# Loop over the different question types
for qa in tqdm(qa_types, desc="qa-types"):
    # Configure BABILong's prompt setup, follows https://github.com/booydar/babilong/blob/main/notebooks/eval_RAG_Llama3.py
    
    prompt_name = [f'{k}_no' if len(prompt_cfg[k]) == 0 else f'{k}_yes' for k in prompt_cfg if k != 'template']
    prompt_name = '_'.join(prompt_name)

    # Loop over the different token sizes
    for token_size in tqdm(text_token_sizes, desc="token-sizes"):
        dataset = adapter.batch_fetch_texts(
            token_size=token_size,
            question_types=[qa],
            text_ids=text_ids
            )[qa]
        
       
        for i, text in enumerate(tqdm(dataset, desc=f"Texts for {qa}-{token_size}k "), start = first_text_id):
            if (qa.value, token_size, i) in processed_ids:
                continue

            text_time_start = time.time()
            if (allow_prompt):
                try:
                    llm_response, time_used_llm = rag(text, prompt_cfg, rag_type)
                    llm_output = llm_response.choices[0].message.content
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
                    with open("RAG_"+rag_type+"_errors", 'a') as error_file:                        
                        error_file.write(error_string)
                    error_file.close()
            else:
                print("LLM Prompt not allowed by own settings")
            time.sleep(0.5) # Ensure never hitting api rate limit
