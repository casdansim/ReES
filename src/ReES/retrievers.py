from abc import ABC, abstractmethod
from typing import Generator, List

from ReES.aggregates import knowledge
from ReES.classification import ClassificationModel
from ReES.knowledge import KnowledgeBase


class Retriever(ABC):

    @abstractmethod
    def retrieve(self, prompt: str) -> Generator[List[knowledge.Entity], None, None]:
        pass


class BasicRetriever(Retriever):

    def __init__(self, knowledge_base: KnowledgeBase, model_classification: ClassificationModel):
        self._knowledge_base = knowledge_base
        self._model_classification = model_classification

    def retrieve(self, prompt: str) -> Generator[List[knowledge.Entity], None, None]:
        entities = self._model_classification.predict_entities(prompt)
        yield self._knowledge_base.get_entities([entity.label for entity in entities])


class RecursiveRetriever(Retriever):

    def __init__(self, knowledge_base: KnowledgeBase, model_classification: ClassificationModel, k: int = 2):
        self._knowledge_base = knowledge_base
        self._basic_retriever = BasicRetriever(knowledge_base, model_classification)
        self._k = k

    def retrieve(self, prompt: str) -> Generator[List[knowledge.Entity], None, None]:
        if self._k <= 0:
            return

        basic_entities = self._basic_retriever.retrieve(prompt)
        basic_entities = [
            entity
            for entities in basic_entities
            for entity in entities
        ]

        yield basic_entities

        labels = set([entity.label for entity in basic_entities])

        for _ in range(1, self._k):
            new_entities = [
                linked_entity
                for linked_entity in self._knowledge_base.get_linked_entities(list(labels))
                if not linked_entity.label in labels
            ]

            yield new_entities

            labels.update([entity.label for entity in new_entities])
