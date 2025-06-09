from abc import ABC, abstractmethod
from typing import List, Dict

from neo4j import GraphDatabase

from ReES.aggregates import knowledge, processing
from ReES.env import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME


class KnowledgeBase(ABC):

    @abstractmethod
    def get_entities(self, labels: List[str]) -> List[knowledge.Entity]:
        """
        Extract all entities with the given labels from the knowledge base.
        :param labels: the labels to filter for.
        :return: all entities with the given labels.
        """
        pass

    @abstractmethod
    def get_unsummarized_chunks(self, labels: List[str]) -> Dict[str, List[knowledge.Chunk]]:
        pass

    @abstractmethod
    def get_linked_entities(self, labels: List[str]) -> List[knowledge.Entity]:
        """
        Retrieves linked entities associated with the provided labels.

        Any implementation of this method must take care to remove duplicates in the returned list.

        :param labels: A list of strings representing the labels for which the
            linked entities are to be retrieved.
        :return: A list of `knowledge.Entity` objects representing
            the entities linked to the provided label.
        """
        pass

    @abstractmethod
    def insert_chunks(
        self,
        chunks: List[processing.Chunk]
    ) -> List[knowledge.Chunk]:
        """
        Insert chunks into knowledge base.

        Any implementation of this method must
            1) keep order of the chunks and transform each chunk in the returned list and
            2) set the sequence number of each processing chunk.
        """
        pass

    @abstractmethod
    def insert_entities(
        self,
        entities: List[processing.Entity],
        links: Dict[str, List[processing.Chunk]],
    ) -> List[knowledge.Entity]:
        """
        Insert entities into knowledge base.

        Any implementation of this method must keep order of the entities and transform each entity in the returned list.
        """
        pass

    @abstractmethod
    def update_summary(self, label: str, summary: str, sequence_number: int, links: List[processing.Entity]):
        """
        Updates the summary of the Entity with the given label.
        :param label: Label of Entity to update.
        :param summary: Updated summary.
        :param sequence_number: An integr representing the last sequence number of processed chunks in the summary.
        :param links: List of Entities linked to the Entity by appearance in the summary.
        """
        pass


class InMemoryKnowledgeBase(KnowledgeBase):

    def __init__(self):
        self._store_chunks: List[knowledge.Chunk] = []
        self._store_entities: Dict[str, knowledge.Entity] = {}
        self._store_chunk_links: Dict[str, List[int]] = {}
        self._store_entity_links: Dict[str, List[str]] = {}

    def get_entities(self, labels: List[str]) -> List[processing.Entity]:
        return [self._store_entities[label] for label in labels if label in self._store_entities]

    def get_linked_entities(self, labels: List[str]) -> List[knowledge.Entity]:
        unique_labels = set([
            linked_label
            for label in labels
            for linked_label in self._store_entity_links[label]
            if linked_label in self._store_entities
        ])

        return [self._store_entities[linked_label] for linked_label in unique_labels]

    def get_unsummarized_chunks(self, labels: List[str]) -> Dict[str, List[knowledge.Chunk]]:
        result = {label: [] for label in labels}

        for label in labels:
            if self._store_chunk_links[label] is None:
                continue

            entity = self._store_entities[label]
            chunk_sequence_numbers = self._store_chunk_links[label]
            chunks = [
                self._store_chunks[sequence_number - 1]
                for sequence_number in chunk_sequence_numbers
                if sequence_number > entity.chunk_sequence
            ]
            result[label] = chunks

        return result

    def insert_chunks(
        self,
        chunks: List[processing.Chunk],
    ) -> List[knowledge.Chunk]:
        new_chunks = []
        for sequence_number, chunk in enumerate(chunks, len(self._store_chunks) + 1):
            chunk.sequence_number = sequence_number
            new_chunks.append(knowledge.Chunk(sequence_number, chunk.text))

        self._store_chunks.extend(new_chunks)
        return new_chunks

    def insert_entities(
        self,
        entities: List[processing.Entity],
        links: Dict[str, List[processing.Chunk]],
    ) -> List[knowledge.Entity]:
        new_entities = []
        for entity in entities:
            new_entities.append(
                self._store_entities[entity.label]
                if entity.label in self._store_entities
                else knowledge.Entity(entity.label, "", 0)
            )

        for entity in new_entities:
            self._store_entities.update({ entity.label: entity })

        self._store_chunk_links.update({
            label: [chunk.sequence_number for chunk in chunks]
            for (label, chunks)
            in links.items()
        })

        return new_entities

    def update_summary(self, label: str, summary: str, sequence_number: int, links: List[processing.Entity]):
        entity = self._store_entities[label]
        entity.summary = summary
        entity.chunk_sequence = sequence_number
        self._store_entity_links[label] = [link.label for link in links]


class Neo4jKnowledgeBase(KnowledgeBase):

    def __init__(self, tenant: str):
        self._tenant = tenant

        self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self._driver.verify_connectivity()

        self._clear_tenant()

    def _clear_tenant(self):
        self._driver.execute_query(
            f"""
            MATCH (:{self._tenant})-[r]->(:{self._tenant})
            DELETE r
            """
        )

        self._driver.execute_query(
            f"""
            MATCH (n:{self._tenant})
            DELETE n
            """
        )

    def get_entities(self, labels: List[str]) -> List[knowledge.Entity]:
        parameters = {
            "labels": labels,
        }

        records, _, _ = self._driver.execute_query(
            f"""
            MATCH (e:{self._tenant}:Entity)
            WHERE e.label IN $labels
            RETURN e.label, e.summary, e.chunk_sequence
            """,
            parameters,
        )

        return [
            knowledge.Entity(record["e.label"], record["e.summary"], record["e.chunk_sequence"])
            for record in records
        ]

    def get_unsummarized_chunks(self, labels: List[str]) -> Dict[str, List[knowledge.Chunk]]:
        parameters = {
            "labels": labels,
        }

        records, _, _ = self._driver.execute_query(
            f"""
            MATCH (e:{self._tenant}:Entity)-[:ORIGINATES_FROM]->(c:{self._tenant}:Chunk)
            WHERE e.label IN $labels AND c.sequence_number > e.chunk_sequence
            RETURN e.label, c.sequence_number, c.text
            """,
            parameters,
        )

        result = {label: [] for label in labels}
        for record in records:
            result[record["e.label"]].append(knowledge.Chunk(record["c.sequence_number"], record["c.text"]))

        return result

    def get_linked_entities(self, labels: List[str]) -> List[knowledge.Entity]:
        parameters = {
            "labels": labels,
        }

        records, _, _ = self._driver.execute_query(
            f"""
            MATCH (e:{self._tenant}:Entity)-[:RELATES_TO]->(r:{self._tenant}:Entity)
            WHERE e.label IN $labels
            RETURN r.label, r.summary, r.chunk_sequence
            """,
            parameters,
        )

        return [
            knowledge.Entity(record["r.label"], record["r.summary"], record["r.chunk_sequence"])
            for record in records
        ]

    def insert_chunks(
        self,
        chunks: List[processing.Chunk],
    ) -> List[knowledge.Chunk]:
        parameters = {
            "chunks": [{
                "index": i,
                "text": chunk.text,
            } for i, chunk in enumerate(chunks)],
        }

        records, _, _ = self._driver.execute_query(
            f"""
            MATCH (c:{self._tenant}:Chunk)
            WITH coalesce(max(c.sequence_number), 0) AS current_max_sequence_number
            UNWIND $chunks AS chunk
            CREATE (c:{self._tenant}:Chunk {{sequence_number: current_max_sequence_number + chunk.index + 1, text: chunk.text}})
            RETURN chunk.index, c.sequence_number, c.text
            """,
            parameters,
        )

        new_chunks = []
        for record in records:
            sequence_number = record["c.sequence_number"]
            chunks[record["chunk.index"]].sequence_number = sequence_number
            new_chunks.append(knowledge.Chunk(sequence_number, record["c.text"]))

        return new_chunks

    def insert_entities(
        self,
        entities: List[processing.Entity],
        links: Dict[str, List[processing.Chunk]],
    ) -> List[knowledge.Entity]:
        parameters = {
            "entities": [{
                "label": entity.label,
                "links": [{
                    "sequence_number": link.sequence_number,
                } for link in links[entity.label]],
            } for entity in entities],
        }

        records, _, _ = self._driver.execute_query(
            f"""
            UNWIND $entities AS entity
            CREATE (e:{self._tenant}:Entity {{label: entity.label, summary: "", chunk_sequence: 0}})
            WITH e, entity.links AS links
            UNWIND links AS link
            MATCH (c:{self._tenant}:Chunk {{sequence_number: link.sequence_number}})
            CREATE (e)-[:ORIGINATES_FROM]->(c)
            WITH DISTINCT e
            RETURN e.label, e.summary, e.chunk_sequence
            """,
            parameters,
        )

        return [
            knowledge.Entity(record["e.label"], record["e.summary"], record["e.chunk_sequence"])
            for record in records
        ]

    def update_summary(self, label: str, summary: str, sequence_number: int, links: List[processing.Entity]):
        parameters = {
            "label": label,
            "links": [{
                "label": link.label,
            } for link in links],
            "sequence_number": sequence_number,
            "summary": summary,
        }

        self._driver.execute_query(
            f"""
            MATCH (e:{self._tenant}:Entity {{label: $label}})
            OPTIONAL MATCH (e)-[r:RELATES_TO]->(:{self._tenant}:Entity)
            DELETE r
            WITH DISTINCT e
            SET e.summary = $summary, e.chunk_sequence = $sequence_number
            WITH e
            UNWIND $links AS link
            MATCH (n:{self._tenant}:Entity {{label: link.label}})
            WHERE e <> n
            CREATE (e)-[:RELATES_TO]->(n)
            """,
            parameters,
        )
