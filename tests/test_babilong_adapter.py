from ReES.babilong import  BABILongAdapter, BABILongQuestionType, BABILongText


class TestBABILongAdapter:

    adapter = BABILongAdapter()

    def test_textWhenFetchingText(self):
        res = self.adapter.fetch_text(1, BABILongQuestionType.qa2)
        assert isinstance(res, BABILongText)

    def test_correctNeedlesWhenFetchingText(self):
        res = self.adapter.fetch_text(1, BABILongQuestionType.qa1)
        assert res.needles == ['John travelled to the hallway.', 'Mary journeyed to the bathroom.', 'Daniel went back to the bathroom.', 'John moved to the bedroom.']

    def test_allTextPropertiesWhenFetchingText(self):
        res = self.adapter.fetch_text(1, BABILongQuestionType.qa4)
        assert res.needles is not None
        assert res.question_type is not None
        assert res.raw_needles is not None
        assert res.target is not None
        assert res.target_question is not None
        assert res.text is not None
        assert res.token_size is not None

    def test_correctTokenSize(self):
        token_size = 4
        res = self.adapter.fetch_text(token_size, BABILongQuestionType.qa3)
        assert res.token_size == token_size

    def test_allContentWhenBatchFetching1k(self):
        res = self.adapter.batch_fetch_texts(1)
        assert len(res[BABILongQuestionType.qa1])  == 100
        assert len(res[BABILongQuestionType.qa2])  == 100
        assert len(res[BABILongQuestionType.qa3])  == 100
        assert len(res[BABILongQuestionType.qa4])  == 100
        assert len(res[BABILongQuestionType.qa5])  == 100
        assert len(res.keys()) == 5
        assert res[BABILongQuestionType.qa1][0] != res[BABILongQuestionType.qa1][53]
        assert res[BABILongQuestionType.qa1][21] != res[BABILongQuestionType.qa3][21]

    def test_batchFetchSpecificIDsAndQuestions(self):
        res = self.adapter.batch_fetch_texts(1,
                                             [BABILongQuestionType.qa2, BABILongQuestionType.qa4],
                                             text_ids=[3, 7, 9])
        assert len(res[BABILongQuestionType.qa2]) == 3
        assert len(res[BABILongQuestionType.qa4]) == 3
        assert len(res.keys()) == 2
