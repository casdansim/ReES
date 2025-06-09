from ReES.chunking import ChunkMergingScenes, default_chunking, default_splitter, find_relevant_chunks
from ReES.classification import GLiNERClassificationModel
from ReES.knowledge import InMemoryKnowledgeBase
from ReES.tokenizer import Tokenizer

from tests import test_utility


class TestChunking:

    def test_chunkContainsMary(self):
        text, _ = test_utility.get_stub_text(token_size=16)
        tokenizer = Tokenizer.from_tiktoken_encoder("gpt-4o-mini")
        classification_model = GLiNERClassificationModel(device="cpu")
        knowledge_base = InMemoryKnowledgeBase()

        # Find chunks and entities
        entities = classification_model.predict_entities(text.text)
        splitter = default_splitter(tokenizer, 256)
        chunks = default_chunking(splitter, text.text)

        # Insert chunks in knowledge base
        knowledge_base.insert_chunks(chunks)

        # Find Mary entity
        mary = next(entity for entity in entities if entity.label == "mary")

        # Get relevant chunks
        relevant_chunks = find_relevant_chunks(chunks, mary)

        # Add mary with links to chunks
        knowledge_base.insert_entities([mary], {mary.label: relevant_chunks})

        # Fetch chunks from knowledge base
        chunks = knowledge_base.get_unsummarized_chunks([mary.label])[mary.label]

        # Merge chunks using scenes
        chunk_merging = ChunkMergingScenes(tokenizer)
        merged_chunks = chunk_merging.merge(chunks)

        # Then Mary has relevant chunks and is in the chunks
        assert len(merged_chunks) > 0

        for chunk in merged_chunks:
            assert "mary" in chunk.text.lower()
