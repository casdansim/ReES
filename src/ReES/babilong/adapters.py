from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset, Dataset, DatasetDict, IterableDatasetDict, IterableDataset


class BABILongQuestionType(Enum):
    qa1 = "qa1"
    qa2 = "qa2"
    qa3 = "qa3"
    qa4 = "qa4"
    qa5 = "qa5"
    qa6 = "qa6"
    qa7 = "qa7"
    qa8 = "qa8"
    qa9 = "qa9"
    qa10 = "qa10"
    qa11 = "qa11"
    qa12 = "qa12"
    qa13 = "qa13"
    qa14 = "qa14"
    qa15 = "qa15"
    qa16 = "qa16"
    qa17 = "qa17"
    qa18 = "qa18"
    qa19 = "qa19"
    qa20 = "qa20"


class BenchmarkText(ABC):
    """
    Abstract Class for full texts to be used on the benchmark.
    """

    @property
    @abstractmethod
    def needles(self) -> List[str]:
        """
        A list of the needles inserted into the text. Needles are split by sentences.
        """
        pass

    @property
    @abstractmethod
    def question_type(self) -> str:
        """
        The BABILong question type (QA1, QA2,..., QA20).
        """
        pass

    @property
    @abstractmethod
    def raw_needles(self) -> str:
        """
        The raw needles as a sequence of sentences.
        """
        pass

    @property
    @abstractmethod
    def target(self) -> str:
        """
        The target value.
        """
        pass

    @property
    @abstractmethod
    def target_question(self) -> str:
        """
        The raw target question asked to the LLM.
        """
        pass

    @property
    @abstractmethod
    def text(self) -> str:
        """
        The raw text to be processed.
        """
        pass

    @property
    @abstractmethod
    def token_size(self) -> int:
        """
        The size of the text in tokens determined by the GPT2 tokeniser.
        """
        pass


class BABILongText(BenchmarkText):
    """
    Holds a single BABILong text of a specific question type, token size, needles and a corresponding target question.
    """

    def __init__(self, text: str, target_question: str, target: str, raw_needles: str, token_size: int, question_type: BABILongQuestionType):
        self._text = text
        self._target_question = target_question
        self._target = target
        self._raw_needles = raw_needles
        self._token_size = token_size
        self._question_type = question_type

        needles = raw_needles.split(". ")
        processed_needles = [f"{needle}." for needle in needles[:-1]]
        processed_needles.append(needles[-1])
        self._needles = processed_needles

    @property
    def needles(self) -> List[str]:
        """
        A list of the needles inserted into the text. Needles are split by sentences.
        """
        return self._needles

    @property
    def question_type(self) -> BABILongQuestionType:
        """
        The BABILong question type (qa1, qa2,..., qa20).
        """
        return self._question_type

    @property
    def raw_needles(self) -> str:
        """
        The raw needles as a sequence of sentences.
        """
        return self._raw_needles

    @property
    def target(self) -> str:
        """
        The target value.
        """
        return self._target

    @property
    def target_question(self) -> str:
        """
        The raw target question to determine accuracy on.
        """
        return self._target_question

    @property
    def text(self) -> str:
        """
        The raw text to be processed.
        """
        return self._text

    @property
    def token_size(self) -> int:
        """
        The size of the text in tokens determined by the GPT2 tokeniser.
        """
        return self._token_size


class TextAdapter(ABC):
    """
    AbstractClass for fetching texts
    """

    @abstractmethod
    def batch_fetch_texts(
        self,
        token_size: int,
        question_types: Optional[List[BABILongQuestionType]] = None,
        text_ids: Optional[List[int]] = None,
    ) -> dict[BABILongQuestionType, List[BABILongText]]:
        """
        Batch fetches BABILong texts of a given token size.
        The needed question types and text ids can be modified if all texts of qa1-qa5 aren't needed.
        Returns a dictionary of {BABILongQuestionType: List[BABILongText]}.
        """

    @abstractmethod
    def fetch_text(
            self,
            token_size: int,
            question_type: BABILongQuestionType,
            text_id: int,
            dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
            needle_datasets: DatasetDict | Dataset | IterableDatasetDict | IterableDataset
    ) -> BABILongText:
        """
        Fetches a single BABILong text based on token size, question type and text id from Hugging Face.
        An optional BABILong dataset can be supplied if batch fetching.
        Returns a BABILongText.
        """


class BABILongTargetBenchmarkAdapter(TextAdapter):
    """
    Adapter to create BABILongText objects from the target BABILong dataset in a folder.
    """

    MAX_I: int = 100

    def __init__(self, dataset_path: Path):
        self._dataset_path = dataset_path
        self._uncreatedQAs = [BABILongQuestionType.qa11, BABILongQuestionType.qa12, BABILongQuestionType.qa13,
                              BABILongQuestionType.qa14, BABILongQuestionType.qa15, BABILongQuestionType.qa16,
                              BABILongQuestionType.qa17, BABILongQuestionType.qa18, BABILongQuestionType.qa19,
                              BABILongQuestionType.qa20]


    def batch_fetch_texts(
            self,
            token_size: int,
            question_types: Optional[List[BABILongQuestionType]] = None,
            text_ids: Optional[List[int]] = None,
    ) -> dict[BABILongQuestionType, List[BABILongText]]:
        """
        Batch fetches BABILong texts of a given token size.
        The needed question types and text ids can be modified if all texts of qa1-qa5 aren't needed.
        Returns a dictionary of {BABILongQuestionType: List[BABILongText]}.
        """        

        if question_types is None:
            question_types = [BABILongQuestionType.qa1, BABILongQuestionType.qa2, BABILongQuestionType.qa3,
                            BABILongQuestionType.qa4, BABILongQuestionType.qa5]

        if text_ids is None:
            text_ids = [i for i in range(0, 100)]

        token_size_string = f"{token_size}k.json"

        texts: dict[BABILongQuestionType, List[BABILongText]] = {}
        for quesion_type in question_types:
            data_file_path = self._dataset_path / Path(f"{quesion_type.value}/") / Path(token_size_string)
            dataset = load_dataset("json", data_files=str(data_file_path))           

            texts[quesion_type] = []

            for text_id in text_ids:
                text = self.fetch_text(token_size, question_type=quesion_type, text_id=text_id, dataset=dataset,
                                       )
                texts[quesion_type].append(text)

        return texts


    def fetch_text(
            self,
            token_size: int,
            question_type: BABILongQuestionType,
            text_id: int = 0,
            dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset = None    ) -> BABILongText:
        """
        Fetches a single BABILong text based on token size, question type and text id from Hugging Face.
        An optional BABILong dataset can be supplied if batch fetching.
        Returns a BABILongText.
        """

        if question_type in self._uncreatedQAs:
            raise Exception(
                "Tried to fetch a BABILong question type where texts are not available. Try one of qa1,...,qa10 instead.")

        if text_id > BABILongAdapter.MAX_ID:
            raise Exception(
                "Tried to index a text with an id which does not exist. Valid range of text id's is [0,...,99]")

        token_size_string = f"{token_size}k"

        if dataset is None:
            dataset = load_dataset(self._endpoint, token_size_string)

        qa_dataset = dataset["train"]

        benchmark_text = BABILongText(
            text = qa_dataset['input'][text_id],
            target_question = qa_dataset['question'][text_id],
            target = qa_dataset['target'][text_id],
            raw_needles = "",
            token_size=token_size,
            question_type=question_type
        )

        return benchmark_text


class BABILongAdapter(TextAdapter):
    """
    Adapter to create BABILongText objects from BABILong's Hugging Face repository.
    """

    MAX_ID: int = 100

    def __init__(self):
        self._endpoint = "RMT-team/babilong"
        self._uncreatedQAs = [BABILongQuestionType.qa11, BABILongQuestionType.qa12, BABILongQuestionType.qa13,
                              BABILongQuestionType.qa14, BABILongQuestionType.qa15, BABILongQuestionType.qa16,
                              BABILongQuestionType.qa17, BABILongQuestionType.qa18, BABILongQuestionType.qa19,
                              BABILongQuestionType.qa20]

    def batch_fetch_texts(
            self,
            token_size: int,
            question_types: Optional[List[BABILongQuestionType]] = None,
            text_ids: Optional[List[int]] = None,
    ) -> dict[BABILongQuestionType, List[BABILongText]]:
        """
        Batch fetches BABILong texts of a given token size.
        The needed question types and text ids can be modified if all texts of qa1-qa5 aren't needed.
        Returns a dictionary of {BABILongQuestionType: List[BABILongText]}.
        """
        if question_types is None:
            question_types = [BABILongQuestionType.qa1, BABILongQuestionType.qa2, BABILongQuestionType.qa3,
                              BABILongQuestionType.qa4, BABILongQuestionType.qa5]

        if text_ids is None:
            text_ids = [i for i in range(0, 100)]

        token_size_string = f"{token_size}k"

        dataset = load_dataset(self._endpoint, token_size_string)
        needle_datasets = load_dataset(self._endpoint, "0k")

        texts: dict[BABILongQuestionType, List[BABILongText]] = {}
        for quesion_type in question_types:
            texts[quesion_type] = []

            for text_id in text_ids:
                text = self.fetch_text(token_size, question_type=quesion_type, text_id=text_id, dataset=dataset,
                                       needle_datasets=needle_datasets)
                texts[quesion_type].append(text)

        return texts

    def fetch_text(
            self,
            token_size: int,
            question_type: BABILongQuestionType,
            text_id: int = 0,
            dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset = None,
            needle_datasets: DatasetDict | Dataset | IterableDatasetDict | IterableDataset = None
    ) -> BABILongText:
        """
        Fetches a single BABILong text based on token size, question type and text id from Hugging Face.
        An optional BABILong dataset can be supplied if batch fetching.
        Returns a BABILongText.
        """

        if question_type in self._uncreatedQAs:
            raise Exception(
                "Tried to fetch a BABILong question type where texts are not available. Try one of qa1,...,qa10 instead.")

        if text_id > BABILongAdapter.MAX_ID:
            raise Exception(
                "Tried to index a text with an id which does not exist. Valid range of text id's is [0,...,99]")

        token_size_string = f"{token_size}k"

        if dataset is None:
            dataset = load_dataset(self._endpoint, token_size_string)

        qa_dataset = dataset[question_type.value]

        # Fetch 0K datasets which contain the list of needles
        if needle_datasets is None:
            needle_datasets = load_dataset(self._endpoint, '0k')

        benchmark_text = BABILongText(
            text=qa_dataset['input'][text_id],
            target_question=qa_dataset['question'][text_id],
            target=qa_dataset['target'][text_id],
            raw_needles=needle_datasets[question_type.value]['input'][text_id],
            token_size=token_size,
            question_type=question_type
        )

        return benchmark_text
