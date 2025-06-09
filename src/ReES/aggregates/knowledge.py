from dataclasses import dataclass


@dataclass
class Chunk:
    sequence_number: int
    text: str


@dataclass
class Entity:
    label: str
    summary: str
    chunk_sequence: int
