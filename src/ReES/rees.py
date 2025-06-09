from abc import ABC, abstractmethod
from asyncio import gather
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

from ReES.aggregates import knowledge
from ReES.aggregates.language import Tokens
from ReES.chunking import default_chunking, default_splitter, find_relevant_chunks, ChunkMerging
from ReES.classification import ClassificationModel
from ReES.knowledge import KnowledgeBase
from ReES.language import LanguageModel
from ReES.prompting import PromptingStrategy
from ReES.retrievers import Retriever
from ReES.tokenizer import Tokenizer


@dataclass
class ReESPrompt:
    prompt: str
    response: str
    time: float
    tokens: Tokens
    entity: Optional[knowledge.Entity]


@dataclass
class ReESAnswer:
    message: str
    prompts: List[ReESPrompt]


@dataclass
class ReESProcess:
    chunks: List[knowledge.Chunk]
    entities: List[knowledge.Entity]
    prompts: List[ReESPrompt]


class ReES(ABC):

    prompt_answer = "You are given a summary of the entity and a question about the entity. Your task is to answer the question. The summary is given between <summary>...</summary> tags, before the question is asked. Answer only with one word!"
    prompt_summarization = "You are given an entity, your previous summary of the entity and a chunk of text with new information on the entity. Your task is to summarize new information on the entity given by the chunk of text with emphasis on relations to other locations, objects and people. Your previous summary is given between <summary>...</summary> tags and the chunk of text is given between <chunk>...</chunk> tags, before an instruction to update your summary of the entity. Answer only with the updated summary and state it definitively!"

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        model_classification: ClassificationModel,
        model_language: LanguageModel,
        prompting_strategy: PromptingStrategy,
        tokenizer: Tokenizer,
        chunk_merging: Optional[ChunkMerging],
        chunk_size: int,
        chunk_overlap: int,
    ):
        self._knowledge_base = knowledge_base
        self._model_classification = model_classification
        self._model_language = model_language
        self._prompting_strategy = prompting_strategy
        self._chunk_merging = chunk_merging

        self._splitter = default_splitter(tokenizer, chunk_size, chunk_overlap)

    @abstractmethod
    async def answer(self, prompt: str) -> ReESAnswer:
        """
        Answers a prompt using its knowledge.
        :param prompt: the prompt to answer.
        :return: a ReESResponse with the answer,
            the time it took to process the prompt,
            and the tokens used by the language model.
        """
        pass

    @abstractmethod
    async def process(self, text: str) -> ReESProcess:
        """
        Processes the text and adds information to its knowledge.
        :param text: the text to process.
        :return: a ReESResponse with an empty answer,
            the time it took to process the text,
            and the tokens used by the language model.
        """
        pass

    def _insert_chunks_and_entities(self, text: str) -> (List[knowledge.Chunk], List[knowledge.Entity]):
        text_chunks = default_chunking(self._splitter, text)
        entities = self._model_classification.predict_entities(text)
        links = {
            entity.label: find_relevant_chunks(text_chunks, entity)
            for entity in entities
        }

        chunks = self._knowledge_base.insert_chunks(text_chunks)
        entities = self._knowledge_base.insert_entities(entities, links)

        return chunks, entities

    async def _answer(
        self,
        prompt: str,
        summaries: List[str],
    ) -> ReESAnswer:
        summary = "\n\n".join(summaries)

        user_prompt = self._prompting_strategy.answer_user_prompt(summary, prompt)

        response = await self._model_language.instruct(
            self._prompting_strategy.answer_system_prompt(),
            user_prompt,
        )

        prompt = ReESPrompt(
            prompt=user_prompt,
            response=response.message,
            time=response.time,
            tokens=response.tokens,
            entity=None,
        )

        return ReESAnswer(
            message=response.message,
            prompts=[prompt],
        )

    async def _summarize_entity(
        self,
        relevant_chunks: List[knowledge.Chunk],
        entity: knowledge.Entity,
        summary: str,
    ) -> ReESAnswer:
        summary = deepcopy(summary)

        prompts = []
        for chunk in relevant_chunks:
            user_prompt = self._prompting_strategy.summarisation_user_prompt(summary, chunk, entity)

            response = await self._model_language.instruct(
                self._prompting_strategy.summarisation_system_prompt(),
                user_prompt
            )

            summary = response.message

            prompts.append(ReESPrompt(
                prompt=user_prompt,
                response=response.message,
                time=response.time,
                tokens=response.tokens,
                entity=entity,
            ))

        return ReESAnswer(
            message=summary,
            prompts=prompts,
        )

    async def _update_summaries(
        self,
        entities: List[knowledge.Entity],
    ) -> List[ReESAnswer]:
        chunks = self._knowledge_base.get_unsummarized_chunks([entity.label for entity in entities])

        async def summarization_task(entity: knowledge.Entity) -> ReESAnswer:
            if self._chunk_merging is None:
                relevant_chunks = chunks[entity.label]
            else:
                relevant_chunks = self._chunk_merging.merge(chunks[entity.label])

            if len(relevant_chunks) == 0:
                return ReESAnswer(
                    message=entity.summary,
                    prompts=[],
                )

            response = await self._summarize_entity(relevant_chunks, entity, entity.summary)
            new_summary = response.message
            sequence_number = relevant_chunks[-1].sequence_number

            linked_entities = self._model_classification.predict_entities(new_summary)
            self._knowledge_base.update_summary(entity.label, new_summary, sequence_number, linked_entities)

            return response

        summarization_tasks = [summarization_task(entity) for entity in entities]
        return await gather(*summarization_tasks)


class DelayedReES(ReES):

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        model_classification: ClassificationModel,
        model_language: LanguageModel,
        prompting_strategy: PromptingStrategy,
        retriever: Retriever,
        tokenizer: Tokenizer,
        chunk_merging: Optional[ChunkMerging],
        chunk_size: int,
        chunk_overlap: int=0,
    ):
        super().__init__(
            knowledge_base,
            model_classification,
            model_language,
            prompting_strategy,
            tokenizer,
            chunk_merging,
            chunk_size,
            chunk_overlap,
        )

        self._retriever = retriever

    async def answer(self, prompt: str) -> ReESAnswer:
        summaries: List[str] = []

        prompts = []
        for entities in self._retriever.retrieve(prompt):
            for response in await self._update_summaries(entities):
                summaries.append(response.message)
                prompts.extend(response.prompts)

        instruction_response = await self._answer(prompt, summaries)
        prompts.extend(instruction_response.prompts)

        return ReESAnswer(
            message=instruction_response.message,
            prompts=prompts,
        )

    async def process(self, text: str) -> ReESProcess:
        chunks, entities = self._insert_chunks_and_entities(text)

        return ReESProcess(
            chunks=chunks,
            entities=entities,
            prompts=[],
        )


class PreprocessingReES(ReES):

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        model_classification: ClassificationModel,
        model_language: LanguageModel,
        prompting_strategy: PromptingStrategy,
        retriever: Retriever,
        tokenizer: Tokenizer,
        chunk_merging: Optional[ChunkMerging],
        chunk_size: int,
        chunk_overlap: int=0,
    ):
        super().__init__(
            knowledge_base,
            model_classification,
            model_language,
            prompting_strategy,
            tokenizer,
            chunk_merging,
            chunk_size,
            chunk_overlap,
        )

        self._retriever = retriever

    async def answer(self, prompt: str) -> ReESAnswer:
        summaries = [
            entity.summary
            for entities in self._retriever.retrieve(prompt)
            for entity in entities
        ]

        return await self._answer(prompt, summaries)

    async def process(self, text: str) -> ReESProcess:
        chunks, entities = self._insert_chunks_and_entities(text)

        responses = await self._update_summaries(entities)
        prompts = [
            prompt
            for response in responses
            for prompt in response.prompts
        ]

        return ReESProcess(
            chunks=chunks,
            entities=entities,
            prompts=prompts,
        )
