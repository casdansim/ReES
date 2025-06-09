from asyncio import gather, run
import math
from pathlib import Path
import time
from typing import List

import pandas
from pandas import DataFrame

from ReES import DelayedReES, ReESPrompt, Tokenizer
from ReES.babilong import BABILongAdapter, BABILongQuestionType, BABILongText
from ReES.chunking import ChunkMergingScenes
from ReES.classification import GLiNERClassificationModel, SuperStringElimination
from ReES.knowledge import InMemoryKnowledgeBase
from ReES.language import OpenAILanguageModel, RateLimiter
from ReES.prompting import BasicPromptingStrategy
from ReES.retrievers import BasicRetriever


path_folder = Path(__file__).parent


# Parameters
name_language = "gpt-4.1-nano-2025-04-14"

max_text_count = 30
max_text_size = 16

gliner_threshold = 0.15

question_types = [
    BABILongQuestionType.qa1,
    BABILongQuestionType.qa2,
    BABILongQuestionType.qa3,
    BABILongQuestionType.qa4,
    BABILongQuestionType.qa5,
]


# Initialization
adapter = BABILongAdapter()

model_classification = GLiNERClassificationModel(SuperStringElimination(), threshold=gliner_threshold)
model_language = RateLimiter(OpenAILanguageModel(name_language), 5_000, 5_000_000)

prompting_strategy = BasicPromptingStrategy()

tokenizer = Tokenizer.from_tiktoken_encoder(name_language)

chunk_merging = ChunkMergingScenes(tokenizer)


# Processing
def compare_answers(target, output):
    return target.lower() == output.strip().lower().strip('".')


async def process_texts(
    token_size: int,
    qa: BABILongQuestionType,
    texts: List[BABILongText],
    chunk_size_offset: int,
    index_offset: int,
    df_prompts: DataFrame,
    df_results: DataFrame,
):
    sizes = [128, 96, 64, 48, 32, 24, 16]
    for chunk_size in [sizes[i] for i in range(chunk_size_offset, len(sizes))]:
        for i, data in enumerate(texts[index_offset:], index_offset + 1):
            knowledge_base = InMemoryKnowledgeBase()
            retriever = BasicRetriever(knowledge_base, model_classification)

            rees = DelayedReES(
                knowledge_base,
                model_classification,
                model_language,
                prompting_strategy,
                retriever,
                tokenizer,
                chunk_merging,
                chunk_size,
            )

            process_begin = time.time()
            process = await rees.process(data.text)
            process_end = time.time()

            answer_begin = time.time()
            answer = await rees.answer(data.target_question)
            answer_end = time.time()

            correct_guess = compare_answers(data.target, answer.message)

            all_prompts: List[ReESPrompt] = []
            all_prompts.extend(process.prompts)
            all_prompts.extend(answer.prompts)

            df_results.loc[len(df_results)] = [
                token_size, # token_size
                chunk_size, # chunk_size
                i, # index
                data.target_question, # question
                data.target, # target
                answer.message, # actual
                correct_guess, # correct_guess
                process_end - process_begin, # process_time
                answer_end - answer_begin, # answer_time
                sum([prompt.time for prompt in all_prompts]), # total_llm_time
                sum([prompt.tokens.input for prompt in all_prompts]), # total_input_tokens
                sum([prompt.tokens.output for prompt in all_prompts]), # total_output_tokens
            ]

            for prompt in all_prompts:
                df_prompts.loc[len(df_prompts)] = [
                    token_size, # token_size
                    chunk_size, # chunk_size
                    i, # index
                    prompt.prompt, # prompt
                    prompt.response, # response
                    prompt.time, # llm_time
                    prompt.tokens.input, # input_tokens
                    prompt.tokens.output, # output_tokens
                    "" if prompt.entity is None else prompt.entity.label, # entity
                ]

            df_prompts.to_csv(path_folder / f"prompts-{qa.value}.csv", index=False)
            df_prompts.to_json(path_folder / f"prompts-{qa.value}.json", orient='records', indent = 4)

            df_results.to_csv(path_folder / f"results-{qa.value}.csv", index=False)
            df_results.to_json(path_folder / f"results-{qa.value}.json", orient='records', indent = 4)

        index_offset = 0


async def process_question_type(qa: BABILongQuestionType):
    path_prompts = path_folder / f"prompts-{qa.value}.csv"
    path_results = path_folder / f"results-{qa.value}.csv"

    if path_prompts.is_file():
        df_prompts = pandas.read_csv(path_prompts)
    else:
        df_prompts = DataFrame(columns=[
            "token_size",
            "chunk_size",
            "index",
            "prompt",
            "response",
            "llm_time",
            "input_tokens",
            "output_tokens",
            "entity",
        ])

    if path_results.is_file():
        df_results = pandas.read_csv(path_results)
    else:
        df_results = DataFrame(columns=[
            "token_size",
            "chunk_size",
            "index",
            "question",
            "target",
            "actual",
            "correct_guess",
            "process_time",
            "answer_time",
            "total_llm_time",
            "total_input_tokens",
            "total_output_tokens",
        ])

    if len(df_results) == 0:
        size_offset = -1
        chunk_size_offset = 0
        text_offset = 0
    else:
        last_result = df_results.loc[len(df_results)-1]
        last_result_token_size = int(last_result["token_size"])

        size_offset = -1 if last_result_token_size == 0 else int(math.log2(last_result_token_size))
        text_offset = int(last_result["index"])

        match int(last_result["chunk_size"]):
            case 128:
                chunk_size_offset = 0
            case 96:
                chunk_size_offset = 1
            case 64:
                chunk_size_offset = 2
            case 48:
                chunk_size_offset = 3
            case 32:
                chunk_size_offset = 4
            case 24:
                chunk_size_offset = 5
            case 16:
                chunk_size_offset = 6
            case _:
                raise Exception("Wrong Chunk Size!")

    for i in range(size_offset, int(math.log2(max_text_size)) + 1):
        token_size = 0 if i == -1 else 2 ** i
        data = adapter.batch_fetch_texts(token_size, [qa], text_ids=list(range(max_text_count)))
        await process_texts(token_size, qa, data[qa], chunk_size_offset, text_offset, df_prompts, df_results)
        chunk_size_offset = 0
        text_offset = 0


async def main():
    tasks = [process_question_type(qa) for qa in question_types]
    await gather(*tasks)


run(main())
