import pickle
from typing import List

from ReES.aggregates.processing import Entity
from ReES.classification import GLiNERClassificationModel
from ReES.language import LanguageModel

from tests import test_utility


class LanguageModelStub(LanguageModel):

    def __init__(self):
        pass

    async def instruct(self, system_prompt, user_prompt):
        return None


class GLiNERStub(GLiNERClassificationModel):

    def __init__(self, labels=None, model_name="urchade/gliner_large-v2.1", device="cpu"):
        super().__init__(labels, model_name, device)

        self._labels = labels
        self._model_name = model_name
        self._device = device

        self.gliner = None
    
    def _batch_predict_entities(self, texts: List[str]):
        return super()._batch_predict_entities(texts)
    
    def predict_entities(self, text: str, file_identifier: str =None) -> List[Entity]:
        filename = test_utility.cache_filename("entities", file_identifier)
        cached_file = test_utility.find_cached_file(filename)

        if cached_file:
            with open(cached_file, 'rb') as input_file:
                entities = pickle.load(input_file)

        else:
            if self.gliner is None:
                super().__init__(self._labels, self._model_name, self._device)

            entities = super().predict_entities(text)
            test_utility.pickle_object(entities, filename)

        return entities
