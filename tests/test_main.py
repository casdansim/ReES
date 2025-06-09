from typing import List

from ReES.aggregates.processing import Chunk
from ReES.chunking import default_splitter, default_chunking, ChunkMergingScenes, find_relevant_chunks
from ReES.tokenizer import Tokenizer

from tests import test_utility
from tests.stubs import GLiNERStub


class TestReES:

    gliner = GLiNERStub()

    llm_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    stub_tokeniser = Tokenizer.from_huggingface_tokenizer(llm_name)

    def write_chunks(self, chunks: List[Chunk], relevant_chunks: List[Chunk], filename = "test_chunks"):
        with open(f'{filename}.txt', 'w') as f:
            for chunk in chunks:
                f.write(chunk.__str__())
                f.write("\n")
        f.close()

        with open(f'{filename}_rcs.txt', 'w') as f:
            for chunk in relevant_chunks:
                f.write(chunk.__str__())
                f.write("\n")
        f.close()

    def test_FixedChunkingStrategySplitChunks(self):
        text, _ = test_utility.get_stub_text()
        splitter = default_splitter(self.stub_tokeniser, 256)
        chunks = default_chunking(splitter, text.text)
        assert len(chunks) == 16

    def test_FixedChunkingStrategyRelevantEntities(self):
        text, file_identifier = test_utility.get_stub_text()
        splitter = default_splitter(self.stub_tokeniser, 256)
        chunks = default_chunking(splitter, text.text)
        
        entities = self.gliner.predict_entities(text.text, file_identifier=file_identifier)
        assert entities[0].label == 'pauline'

        rcs = find_relevant_chunks(chunks, entities[0])

        assert len(rcs) == 16

    def test_SceneChunkingStrategyRelevantEntities(self):
        # Should output one large chunk for 'pauline'
        text, file_identifier = test_utility.get_stub_text()
        splitter = default_splitter(self.stub_tokeniser, 256)
        chunks = default_chunking(splitter, text.text)

        entities = self.gliner.predict_entities(text.text, file_identifier=file_identifier)
        rcs = find_relevant_chunks(chunks, entities[0])

        chunk_merging = ChunkMergingScenes(self.stub_tokeniser)
        scene_chunks = chunk_merging.merge(test_utility.convert_chunks(rcs))

        assert len(scene_chunks) == 1
        assert len(scene_chunks[0].text) == 14465
    
    def test_SceneChunkingStrategyRelevantEntitiesLowMaxChunkSize(self):
        # Should output two large chunks for 'pauline'
        text, file_identifier = test_utility.get_stub_text()
        splitter = default_splitter(self.stub_tokeniser, 256)
        chunks = default_chunking(splitter, text.text)

        entities = self.gliner.predict_entities(text.text, file_identifier=file_identifier)
        rcs = find_relevant_chunks(chunks, entities[0])

        chunk_merging = ChunkMergingScenes(self.stub_tokeniser, max_chunk_length=2048)
        scene_chunks = chunk_merging.merge(test_utility.convert_chunks(rcs))

        self.write_chunks(chunks, rcs)

        assert len(scene_chunks) == 2
        assert len(scene_chunks[0].text) == 7617
        assert len(scene_chunks[1].text) == 6848

    def test_SceneChunkingRelevantEntities32k(self):
        text, file_identifier = test_utility.get_stub_text(token_size=32)
        splitter = default_splitter(self.stub_tokeniser, 256)
        chunks = default_chunking(splitter, text.text)

        entities = self.gliner.predict_entities(text.text, file_identifier=file_identifier)
        rcs = find_relevant_chunks(chunks, entities[0])

        fixed_chunks = test_utility.convert_chunks(rcs)

        chunk_merging = ChunkMergingScenes(self.stub_tokeniser)
        scene_chunks = chunk_merging.merge(test_utility.convert_chunks(rcs))

        assert len(scene_chunks) < len(fixed_chunks)
        assert len(scene_chunks) != 0

    def test_SceneChunkingOutputsAllChunks(self):
        # Should output one relevant chunk for 'good old lady'
        text, file_identifier = test_utility.get_stub_text(token_size=16)
        splitter = default_splitter(self.stub_tokeniser, 256)
        chunks = default_chunking(splitter, text.text)

        entities = self.gliner.predict_entities(text.text, file_identifier=file_identifier)
        rcs = find_relevant_chunks(chunks, entities[0])

        chunk_merging = ChunkMergingScenes(self.stub_tokeniser)
        scene_chunks = chunk_merging.merge(test_utility.convert_chunks(rcs))

        assert len(scene_chunks) == 1

    def test_SceneChunkingOverlap(self):
        # Pauline (entities[0]) is in all checks except one in the middle
        # Text is short, so it should create two relevant chunks
        text, file_identifier = test_utility.get_stub_text()
        splitter = default_splitter(self.stub_tokeniser, 256, chunk_overlap=128)
        chunks = default_chunking(splitter, text.text)

        entities = self.gliner.predict_entities(text.text, file_identifier=file_identifier)
        rcs = find_relevant_chunks(chunks, entities[0])

        chunk_merging = ChunkMergingScenes(self.stub_tokeniser, max_chunk_length=1830)
        scene_chunks = chunk_merging.merge(test_utility.convert_chunks(rcs))

        length_of_first_chunk = self.stub_tokeniser.len(scene_chunks[0].text)
        length_of_second_chunk = self.stub_tokeniser.len(scene_chunks[1].text)
        length_of_all_chunks = sum(map(lambda x: self.stub_tokeniser.len(x.text), chunks))

        assert len(scene_chunks) == 2
        assert length_of_all_chunks * 0.4 < length_of_first_chunk + length_of_second_chunk < length_of_all_chunks * 0.6

    def test_defaultChunkingIndicesNoOverlap(self):
        text, file_identifier = test_utility.get_stub_text()
        splitter = default_splitter(self.stub_tokeniser, 256)
        chunks = default_chunking(splitter, text.text)

        for i, chunk in enumerate(chunks):
            assert chunk.start >= 0
            if i > 0:
                assert chunks[i].start > chunks[i-1].start
                assert chunks[i].end > chunks[i-1].end
    
    def test_defaultChunkingIndicesWithOverlap(self):
        text, file_identifier = test_utility.get_stub_text()
        splitter = default_splitter(self.stub_tokeniser, 256, chunk_overlap=128)
        chunks = default_chunking(splitter, text.text)

        for i, chunk in enumerate(chunks):
            assert chunk.start >= 0
            if i > 0:
                assert chunks[i].start > chunks[i-1].start
                assert chunks[i].end > chunks[i-1].end


if __name__ == "__main__":
    test_suite = TestReES()
    # test_suite.test_FixedChunkingStrategySplitChunks()
    # test_suite.test_FixedChunkingStrategyRelevantEntities()
    # test_suite.test_FixedChunkingSceneChunkingSimilarSplits()
    test_suite.test_SceneChunkingStrategyRelevantEntities()
    # test_suite.test_SceneChunkingRelevantEntities32k()
    # test_suite.test_BasicChunk()
    # test_suite.test_SceneChunkingOverlap()
    # test_suite.test_defaultChunkingIndicesNoOverlap()
    # test_suite.test_defaultChunkingIndicesWithOverlap()
    # test_suite.test_SceneRelevantChunksFor128k()
