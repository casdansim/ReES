from abc import ABC, abstractmethod
from itertools import batched
from typing import Dict, List, Optional

from gliner import GLiNER

from ReES.aggregates.processing import Entity, Location
from ReES.chunking import default_splitter
from ReES.env import CLASSIFICATION_BATCH
from ReES.tokenizer import Tokenizer


class ClassificationModel(ABC):

    @abstractmethod
    def predict_entities(self, text: str) -> List[Entity]:
        pass


class PredictionStrategy(ABC):

    @abstractmethod
    def enable_flat_ner(self) -> bool:
        pass

    @abstractmethod
    def post_process_batches(self, batches: List[List[Dict[str, str]]]) -> List[List[Dict[str, str]]]:
        pass


class BasicFlattening(PredictionStrategy):

    def enable_flat_ner(self) -> bool:
        return True

    def post_process_batches(self, batches: List[List[Dict[str, str]]]) -> List[List[Dict[str, str]]]:
        return batches


class SuperStringElimination(PredictionStrategy):

    def enable_flat_ner(self) -> bool:
        return False

    def post_process_batches(self, batches: List[List[Dict[str, str]]]) -> List[List[Dict[str, str]]]:
        return [[
            entity
            for i, entity
            in enumerate(batch)
            if i == len(batch) - 1
            or not (entity["start"] <= batch[i + 1]["start"] and batch[i + 1]["end"] <= entity["end"])
        ] for batch in batches]


class GLiNERClassificationModel(ClassificationModel):

    _batch_size: int = CLASSIFICATION_BATCH
    _filtered_tokens: List[str] = [
        "i", "you", "he", "she", "it", "we", "they", "one", "me", "him", "her", "us", "them", "my",
        "your", "his", "her", "its", "our", "their", "ones", "myself", "yourself", "himself", "herself",
        "itself", "ourselves", "themselves", "oneself",
    ]

    def __init__(
        self,
        strategy: PredictionStrategy,
        labels: Optional[List[str]] = None,
        model_name: str = "urchade/gliner_large-v2.1",
        threshold=0.15,
        device = "cuda",
    ):
        if labels is None:
            labels = ["location", "object", "person"]

        self._strategy = strategy
        self._labels = labels
        self._threshold = threshold

        self._model = GLiNER.from_pretrained(
            model_name,
        )
        self._model.to(device)
        self._tokenizer = Tokenizer.from_huggingface_tokenizer(self._model.config.model_name)

    def _batch_predict_entities(self, texts: List[str]) -> List[List[Dict[str, str]]]:
        # Batch Predict Without Flattening
        batches: List[List[Dict[str, str]]] = self._model.batch_predict_entities(
            texts,
            self._labels,
            self._strategy.enable_flat_ner(),
            self._threshold,
            False,
        )

        # Substring Elimination Strategy
        batches = [sorted(batch, key=lambda e: (e["start"], -e["end"])) for batch in batches]

        return self._strategy.post_process_batches(batches)

    def predict_entities(self, text: str) -> List[Entity]:
        splitter = default_splitter(self._tokenizer, self._model.config.max_len)
        chunks = splitter.split_text(text)

        # Predict Entities
        location_offset = 0
        predicted_entities = []
        for batch in batched(chunks, GLiNERClassificationModel._batch_size):
            for i, entities in enumerate(self._batch_predict_entities(list(batch))):
                for entity in entities:
                    entity["start"] += location_offset
                    entity["end"] += location_offset
                    predicted_entities.append(entity)

                location_offset += len(batch[i])

        # Batch All Duplicate Entities
        distinct_entities: dict[str, Entity] = {}
        for entity in predicted_entities:
            text = entity["text"].lower().strip()
            text = text.replace("the ", "").strip() if text.startswith("the ") else text

            if text in GLiNERClassificationModel._filtered_tokens:
                continue

            if text not in distinct_entities:
                distinct_entities[text] = Entity(text, [], [])

            distinct_entities[text].locations.append(Location(int(entity["start"])))

        return list(distinct_entities.values())
