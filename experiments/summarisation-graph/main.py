from asyncio import run
from pathlib import Path

from ReES import PreprocessingReES, Tokenizer
from ReES.babilong import BABILongQuestionType, BABILongTargetBenchmarkAdapter
from ReES.chunking import ChunkMergingScenes
from ReES.classification import GLiNERClassificationModel, SuperStringElimination
from ReES.knowledge import Neo4jKnowledgeBase
from ReES.language import OpenAILanguageModel, RateLimiter
from ReES.prompting import BasicPromptingStrategy
from ReES.retrievers import BasicRetriever


file_path = Path(__file__)

# Parameters
name_language = "gpt-4.1-nano-2025-04-14"

chunk_size = 64
question_types = [BABILongQuestionType.qa5]
text_count = 1
text_offset = 0
text_size = 1

# Initialization
adapter = BABILongTargetBenchmarkAdapter(file_path.parent / "../.." / "target_dataset")

model_classification = GLiNERClassificationModel(SuperStringElimination(), device="cpu")
model_language = RateLimiter(OpenAILanguageModel(name_language), 50, 20_000)

tokenizer = Tokenizer.from_tiktoken_encoder(name_language)

prompting_strategy = BasicPromptingStrategy()
chunk_merging = ChunkMergingScenes(tokenizer)


# Processing
async def main():
    dataset = adapter.batch_fetch_texts(
        text_size,
        question_types,
        list(range(text_offset, text_offset + text_count))
    )

    for qa in question_types:
        for i, data in enumerate(dataset[qa], text_offset + 1):
            print(f"--- {i}) {data.target_question} ---", end="\n\n")

            knowledge_base = Neo4jKnowledgeBase(f"TMP".upper())
            retriever = BasicRetriever(knowledge_base, model_classification)

            rees = PreprocessingReES(
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


run(main())
