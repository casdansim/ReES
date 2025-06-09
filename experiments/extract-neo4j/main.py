from pathlib import Path

from neo4j import GraphDatabase

import pandas

from ReES.aggregates import knowledge
from ReES.babilong import BABILongQuestionType
from ReES.chunking import ChunkMergingScenes
from ReES.env import NEO4J_PASSWORD, NEO4J_USERNAME, NEO4J_URI
from ReES.tokenizer import Tokenizer


path_folder = Path(__file__).parent

# Labels for different experiments:
# Main experiment:      'L'
# SSE Ablation Study:   'SSE'
# RR Ablation Study:    'RR'
tenant_identifier = "L"

name_language = "gpt-4.1-nano-2025-04-14"

dataframe = pandas.DataFrame(columns=[
    "token_size",
    "qa",
    "index",
    "entities",
    "n_relevant_chunks",
    "n_merged_chunks",
    "is_summarised",
])

tokenizer = Tokenizer.from_tiktoken_encoder(name_language)
merger = ChunkMergingScenes(tokenizer)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
driver.verify_connectivity()

start_index = 21
end_index = 51

for qa in [
    BABILongQuestionType.qa1,
    BABILongQuestionType.qa2,
    BABILongQuestionType.qa3,
    BABILongQuestionType.qa4,
    BABILongQuestionType.qa5,
]:
    for i in range(-1, 5):
        token_size = 0 if i == -1 else 2 ** i

        for j in range(start_index, end_index):
            records_entities, _, _ = driver.execute_query(
                f"""
                MATCH (e:{tenant_identifier}{token_size}K{qa.value}Q{j}:Entity)
                RETURN e.label, e.summary
                """
            )

            entities = {
                record["e.label"]: {
                    "summary": record["e.summary"],
                    "relevant_chunks": [],
                }
                for record in records_entities
            }

            records_chunks, _, _ = driver.execute_query(
                f"""
                MATCH (e:{tenant_identifier}{token_size}K{qa.value}Q{j}:Entity)-[:ORIGINATES_FROM]->(c:{tenant_identifier}{token_size}K{qa.value}Q{j}:Chunk)
                RETURN e.label, c.sequence_number, c.text
                """
            )

            for record in records_chunks:
                chunk = knowledge.Chunk(
                    sequence_number=record["c.sequence_number"],
                    text=record["c.text"],
                )

                entities[record["e.label"]]["relevant_chunks"].append(chunk)

            for label, data in entities.items():
                merged_chunks = merger.merge(data["relevant_chunks"])

                dataframe.loc[len(dataframe)] = [
                    token_size, # token_size
                    qa, # qa
                    j, # index
                    label, # entities
                    len(data["relevant_chunks"]), # n_relevant_chunks
                    len(merged_chunks), # n_merged_chunks
                    len(data["summary"]) > 0, # is_summarised
                ]

        dataframe.to_csv(path_folder / f"results-{tenant_identifier}.csv", index=False)
