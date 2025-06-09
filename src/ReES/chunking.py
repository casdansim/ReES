from abc import ABC, abstractmethod
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ReES.aggregates import knowledge, processing
from ReES.tokenizer import Tokenizer


def default_chunking(splitter: RecursiveCharacterTextSplitter, text: str) -> List[processing.Chunk]:
    chunks = splitter.split_text(text)

    result = []

    previous_chunk_start = 0
    for i, chunk in enumerate(chunks):
        start = text.find(chunk, previous_chunk_start)
        length = len(chunk)
        end = start + length - 1

        result.append(processing.Chunk(end=end, start=start, text=chunk))
        previous_chunk_start = start

    return result


def default_splitter(tokenizer: Tokenizer, chunk_size: int, chunk_overlap: int = 0)\
    -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        length_function=tokenizer.len,
        separators=[r"[\.|!|?]+\"?\s+"],
        keep_separator="end",
        is_separator_regex=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strip_whitespace=False,
    )


def find_relevant_chunks(chunks: List[processing.Chunk], entity: processing.Entity) -> List[processing.Chunk]:
    locations = iter(entity.locations)
    relevant_chunks = []

    location = next(locations, None)
    for i in range(len(chunks)):
        while location is not None:
            if chunks[i].end < location.start:
                break

            if chunks[i].start <= location.start:
                relevant_chunks.append(chunks[i])
                break

            location = next(locations, None)

    return relevant_chunks


class ChunkMerging(ABC):

    @abstractmethod
    def merge(self, chunks: List[knowledge.Chunk]) -> List[knowledge.Chunk]:
        """
        Given a list of chunks, merges them into a new list of chunks.
        """
        pass


class ChunkMergingScenes(ChunkMerging):

    def __init__(self, tokenizer: Tokenizer, max_chunk_length: int = 8192):
        self._tokenizer = tokenizer
        self._max_chunk_length = max_chunk_length

    @staticmethod
    def _concatenate_prune_overlap(current: knowledge.Chunk, incoming: knowledge.Chunk) -> knowledge.Chunk:
        """
        Return A and B concatenated, but if A ends with some string X
        that B starts with, prune that overlap.
        Inherits B's sequence number to continuing merging of the following chunks.
        """
        max_overlap = min(len(current.text), len(incoming.text))

        for i in range(max_overlap, 0, -1):
            if current.text.endswith(incoming.text[:i]):
                return knowledge.Chunk(
                    incoming.sequence_number,
                    current.text + incoming.text[i:]
                )

        return knowledge.Chunk(incoming.sequence_number, current.text + incoming.text)

    def merge(self, chunks: List[knowledge.Chunk]) -> List[knowledge.Chunk]:
        """
        A chunking strategy with dynamic chunk size. Uses the set chunk hyperparameters, but if an
        entity is found in two subsequent chunks, then they are merged into one larger chunk.
        Chunks can be merged up until some maximum chunk size (given in tokens).
        """
        result = []

        current_chunk = None
        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk
                continue

            # If the chunks are not in sequence, then they should not be merged.
            if current_chunk.sequence_number + 1 != chunk.sequence_number:
                result.append(current_chunk)
                current_chunk = chunk
                continue

            # Merges current_chunk and chunk in next_chunk_candidate pruning any overlap.
            next_chunk_candidate = ChunkMergingScenes._concatenate_prune_overlap(current_chunk, chunk)

            # If the new candidate is larger than the threshold, add old chunk to list and continue from new.
            if self._tokenizer.len(next_chunk_candidate.text) > self._max_chunk_length:
                result.append(current_chunk)
                current_chunk = chunk
                continue

            # Continue merging with the following chunks.
            current_chunk = next_chunk_candidate

        if current_chunk is not None:
            result.append(current_chunk)

        return result
