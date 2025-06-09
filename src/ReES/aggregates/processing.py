from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    start: int
    end: int
    text: str
    sequence_number: Optional[int] = None


@dataclass
class Location:
    start: int


@dataclass
class Entity:
    label: str
    locations: List[Location]
    relevant_chunks: List[Chunk]
