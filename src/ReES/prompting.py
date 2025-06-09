from abc import ABC, abstractmethod

from ReES.aggregates.processing import Chunk, Entity


class PromptingStrategy(ABC):
    """
    A component for returning prompt-templates for specific LLM tasks.
    """

    @abstractmethod
    def answer_system_prompt(self) -> str:
        """
        Returns the system prompt to be used for answering a BABILong benchmark question.
        """
        pass

    @abstractmethod
    def answer_user_prompt(self, summary: str, question: str) -> str:
        """
        Input the summary used to answer the question, as well as the question itself.
        Returns the user prompt which can be forwarded to the language model.
        """
        pass

    @abstractmethod
    def summarisation_system_prompt(self) -> str:
        """
        Returns the system prompt for summarising entities.
        """
        pass

    @abstractmethod
    def summarisation_user_prompt(self, summary: str, chunk: Chunk, entity: Entity) -> str:
        """
        Used for creating a prompt to the LLM for updating an entity's summary based on a new chunk and the existing information about the entity, including its summary.
        Returns a user prompt which can be forwarded to the language model.
        """
        pass


class BasicPromptingStrategy(PromptingStrategy):

    def __init__(self):
        self._summarisation_system_prompt = "You are given an entity, your previous summary of the entity and a chunk of text with new information on the entity. Your task is to summarize new information on the entity given by the chunk of text with emphasis on relations to other locations, objects and people. Your previous summary is given between <summary>...</summary> tags and the chunk of text is given between <chunk>...</chunk> tags, before an instruction to update your summary of the entity. Answer only with the updated summary and state it definitively!"
        self._answer_system_prompt = "You are given a summary of the entity and a question about the entity. Your task is to answer the question. The summary is given between <summary>...</summary> tags, before the question is asked. Answer only with one word!"

    def answer_system_prompt(self) -> str:
        """
        Returns the system prompt to be used for answering a BABILong benchmark question.
        """
        return self._answer_system_prompt

    def answer_user_prompt(self, summary: str, question: str) -> str:
        """
        Input the summary used to answer the question, as well as the question itself.
        Returns the user prompt which can be forwarded to the language model.
        """
        return f"<summary>\n{summary}\n</summary>\n{question}"

    def summarisation_system_prompt(self) -> str:
        """
        Returns the system prompt for summarising entities.
        """
        return self._summarisation_system_prompt

    def summarisation_user_prompt(self, summary: str, chunk: Chunk, entity: Entity) -> str:
        """
        Used for creating a prompt to the LLM for updating an entity's summary based on a new chunk and the existing information about the entity, including its summary.
        Returns a user prompt which can be forwarded to the language model.
        """
        return f"<summary>\n{summary}\n</summary>\n<chunk>\n{chunk.text}\n</chunk>\nUpdate the summary of `{entity.label}`."
