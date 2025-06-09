from asyncio import run

from ReES import DelayedReES
from ReES.babilong import BABILongAdapter, BABILongQuestionType
from ReES.chunking import ChunkMergingScenes
from ReES.classification import GLiNERClassificationModel, SuperStringElimination
from ReES.knowledge import InMemoryKnowledgeBase, Neo4jKnowledgeBase
from ReES.language import LocalLanguageModel, OpenRouterModel, RateLimiter
from ReES.prompting import BasicPromptingStrategy
from ReES.retrievers import BasicRetriever, RecursiveRetriever
from ReES.tokenizer import Tokenizer


# Parameters
chunk_size = 128
# name_language = "qwen/qwen3-235b-a22b:free"
name_language = "Qwen/Qwen3-8B-AWQ"
question_types = [BABILongQuestionType.qa1]
text_count = 1
text_offset = 0
text_size = 8

# Initialization
adapter = BABILongAdapter()

model_classification = GLiNERClassificationModel(SuperStringElimination(), device="cpu")
model_language = LocalLanguageModel(name_language)

tokenizer = Tokenizer.from_huggingface_tokenizer(name_language)

prompting_strategy = BasicPromptingStrategy()
chunk_merging = ChunkMergingScenes(tokenizer)

# Processing
dataset = adapter.batch_fetch_texts(
    text_size,
    question_types,
    list(range(text_offset, text_offset + text_count))
)

async def main():
    for qa in question_types:
        for i, data in enumerate(dataset[qa], 1):
            print(f"--- {i}) {data.target_question} ---", end="\n\n")

            knowledge_base = InMemoryKnowledgeBase()
            retriever = RecursiveRetriever(knowledge_base, model_classification)

            rees = DelayedReES(
                knowledge_base,
                model_classification,
                model_language,
                prompting_strategy,
                retriever,
                tokenizer,
                chunk_merging,
                chunk_size,
                chunk_overlap=chunk_size // 4,
            )

            await rees.process(data.text)
            answer = await rees.answer(data.target_question)
            print(f"Expected {data.target}, got {answer.message}.")
            print(f"Prompts: {answer.prompts}")
            print(f"Time spent: {sum([prompt.time for prompt in answer.prompts])}")
            print(f"Tokens spent: {sum([prompt.tokens.input for prompt in answer.prompts])} input, {sum([prompt.tokens.output for prompt in answer.prompts])} output")
            print()


run(main())
