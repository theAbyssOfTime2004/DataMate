

def test_initialism_value_matches_the_words_it_stands_for() -> None:
    from datamate.query.grounding import QuestionText

    assert QuestionText("đơn bán trên sàn thương mại điện tử").mentions_value("Sàn TMĐT")
    assert QuestionText("don ban tren san thuong mai dien tu").mentions_value("Sàn TMĐT")
    assert not QuestionText("đơn bán trên sàn").mentions_value("Sàn TMĐT")
    assert not QuestionText("thu tien mat dau tuan").mentions_value("Sàn TMĐT")
