"""make_quiz の通常時と該当用語がない場合を、APIを呼ばずに確認する。"""

import json

import pytest

from app import tools


FAKE_GLOSSARY = [
    {"term": "トークン", "definition": "テキストを分割した最小単位", "lecture": 3},
    {"term": "埋め込み", "definition": "語を数値ベクトルで表したもの", "lecture": 3},
    {"term": "勾配降下法", "definition": "損失が下がる方向にパラメータを更新する手法", "lecture": 1},
]


def test_make_quiz_returns_question_and_answer(monkeypatch):
    monkeypatch.setattr(tools, "load_glossary", lambda: FAKE_GLOSSARY)

    raw = tools.make_quiz(3)
    assert "意味は？" in raw
    quiz = json.loads(raw)
    assert set(quiz) == {"question", "answer"}
    lecture3 = {e["term"]: e["definition"] for e in FAKE_GLOSSARY if e["lecture"] == 3}
    term = quiz["question"].removeprefix("「").removesuffix("」の意味は？")
    assert term in lecture3
    assert quiz["answer"] == lecture3[term]


@pytest.mark.parametrize("lecture", [0, -1, 99])
def test_make_quiz_no_terms(monkeypatch, lecture):
    monkeypatch.setattr(tools, "load_glossary", lambda: FAKE_GLOSSARY)

    result = tools.make_quiz(lecture)
    assert "見つかりませんでした" in result
