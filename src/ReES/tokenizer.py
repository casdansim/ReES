from collections.abc import Callable
from typing import List

import tiktoken

from transformers import AutoTokenizer


class Tokenizer:

    def __init__(self, encode_function: Callable[[str], List[int]]):
        self._encode = encode_function

    @staticmethod
    def from_huggingface_tokenizer(model_name: str):
        encoding = AutoTokenizer.from_pretrained(model_name)
        return Tokenizer(encode_function=encoding.encode)

    @staticmethod
    def from_tiktoken_encoder(model_name: str):
        tiktoken_name = "gpt-4o" if model_name.startswith("gpt-4.1") else model_name
        encoding = tiktoken.encoding_for_model(tiktoken_name)
        return Tokenizer(encode_function=encoding.encode)

    def encode(self, text: str) -> List[int]:
        return self._encode(text)

    def len(self, text: str) -> int:
        return len(self.encode(text))


class HuggingFaceTokenizer(Tokenizer):

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        super().__init__(self.tokenizer.encode)
