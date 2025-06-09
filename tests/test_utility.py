from pathlib import Path
import pickle
from typing import Any, List

from ReES.aggregates import knowledge, processing
from ReES.babilong.adapters import BABILongQuestionType, BABILongText, BABILongAdapter


script_path = Path(__file__).parent.resolve()


def convert_chunks(chunks: List[processing.Chunk]) -> List[knowledge.Chunk]:
    return [knowledge.Chunk(i, chunk.text) for i, chunk in enumerate(chunks)]


def get_stub_text(
    token_size: int = 4,
    question_type: BABILongQuestionType = BABILongQuestionType.qa1
) -> tuple[BABILongText, str]:
    file_identifer = f"{token_size}_{question_type}"  # File identifier for tokensize and question type to easily identify cached results
    filename = cache_filename("text", file_identifer)

    cached_file = find_cached_file(filename)
    if cached_file:
        return load_pickle(filename), file_identifer

    # If text is not cached then retrieve it and cache it
    adapter = BABILongAdapter()
    text = adapter.fetch_text(token_size=token_size, question_type=question_type)
    pickle_object(text, filename)

    return text, file_identifer


def pickle_object(obj: Any, filename: str):
    with open(script_path / f"{filename}.pkl", 'wb') as output_file:
        pickle.dump(obj, output_file, pickle.HIGHEST_PROTOCOL)

    output_file.close()


def load_pickle(filename: str) -> Any:
    obj = None

    try:
        with open(script_path / f"{filename}.pkl", 'rb') as input_file:
            obj = pickle.load(input_file)
    except FileNotFoundError:
        with open(script_path / f"{filename}.pkl", 'rb') as input_file:
            obj = pickle.load(input_file)

    return obj


def find_cached_file(filename: str) -> Any:
    pkl_filename = script_path / f"{filename}.pkl"

    if pkl_filename.is_file():
        return pkl_filename

    return None


def cache_filename(cache_type: str, file_identifier: str) -> str:
    return f"stub_{cache_type}_{file_identifier}"
