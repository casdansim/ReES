from asyncio import run
import math
import time

import pandas

from ReES import DelayedReES
from ReES.babilong import BABILongAdapter, BABILongQuestionType
from ReES.chunking import ChunkMergingScenes
from ReES.classification import GLiNERClassificationModel
from ReES.knowledge import InMemoryKnowledgeBase
from ReES.language import OpenAILanguageModel
from ReES.prompting import BasicPromptingStrategy
from ReES.retrievers import BasicRetriever


# Parameters
name_language = "gpt-4.1-nano-2025-04-14"
question_types = [
    BABILongQuestionType.qa1,
    BABILongQuestionType.qa2,
    BABILongQuestionType.qa3,
    BABILongQuestionType.qa4,
    BABILongQuestionType.qa5,
]
text_count = 20
text_offset = 0

# Initialization
adapter = BABILongAdapter()

model_language = OpenAILanguageModel(name_language)

tokenizer = model_language._tokenizer

prompting_strategy = BasicPromptingStrategy()
chunk_merging = ChunkMergingScenes(tokenizer)


# Processing
def compare_answers(target, output):
    return target.lower() == output.strip().lower().strip('".')


async def gliner_threshold():
    # GLiNER Threshold
    chunk_size = 128
    chunk_overlap = 32

    dataframe = pandas.DataFrame(columns=[
        "text_size",
        "qa",
        "index",
        "threshold",
        "entities",
        "question",
        "target",
        "actual",
        "correct_guess",
        "process_time",
        "answer_time",
        "total_llm_time",
        "prompts",
        "input_tokens",
        "output_tokens",
    ])

    path = f"threshold-{int(math.floor(time.time()))}"

    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25]:
        model_classification = GLiNERClassificationModel(threshold=threshold)

        for i in range(-1, 5):
            text_size = 0 if i == -1 else 2 ** i

            dataset = adapter.batch_fetch_texts(
                text_size,
                question_types,
                list(range(text_offset, text_offset + text_count))
            )

            for qa in question_types:
                for j, data in enumerate(dataset[qa], text_offset + 1):
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
                        chunk_overlap=chunk_overlap,
                    )

                    process_begin = time.time()
                    process = await rees.process(data.text)
                    process_end = time.time()

                    answer_begin = time.time()
                    answer = await rees.answer(data.target_question)
                    answer_end = time.time()

                    result = compare_answers(data.target, answer.message)

                    dataframe.loc[len(dataframe)] = [
                        text_size,
                        qa.value,
                        j,
                        threshold,
                        len(process.entities),
                        data.target_question,
                        data.target,
                        answer.message,
                        result,
                        math.ceil(process_end - process_begin),
                        math.ceil(answer_end - answer_begin),
                        math.ceil(process.time_language + answer.time_language),
                        process.prompts + answer.prompts,
                        process.tokens.input + answer.tokens.input,
                        process.tokens.output + answer.tokens.output,
                    ]

                    dataframe.to_csv(f"{path}.csv", index=False)
                    dataframe.to_json(f"{path}.json", orient='records', indent=4)


async def main():
    await gliner_threshold()


run(main())
