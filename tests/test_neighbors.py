"""第5回 宿題（実装）: 検索結果に前後ページを足す処理のテスト。

search() を偽物に差し替えるので、APIキーも検索DBも使わない。ページ一覧は本物の教材JSONを使う。
"""

import json

from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import retrieval
from app.agent import agent


def _hits(*page_ids):
    pages = {page["page_id"]: page for page in retrieval.load_pages()}
    return [{**pages[page_id], "distance": 0.1 * rank} for rank, page_id in enumerate(page_ids, 1)]


def _summary(results):
    return [(r["page_id"], r["role"], r.get("neighbor_of")) for r in results]


def test_window_0_returns_search_results_unchanged(monkeypatch):
    hits = _hits("lec02-main-p052")
    monkeypatch.setattr(retrieval, "search", lambda query: hits)
    assert retrieval.search_with_neighbors("LoRA", window=0) == hits


def test_neighbors_follow_each_hit_without_duplicates(monkeypatch):
    # p.51 は p.52 の前ページでもあるが、2位の当たりなので当たりとして1回だけ返す
    monkeypatch.setattr(retrieval, "search", lambda query: _hits("lec02-main-p052", "lec02-main-p051"))
    results = retrieval.search_with_neighbors("LoRA", window=1)
    assert _summary(results) == [
        ("lec02-main-p052", "hit", None),
        ("lec02-main-p053", "neighbor", "lec02-main-p052"),
        ("lec02-main-p051", "hit", None),
        ("lec02-main-p050", "neighbor", "lec02-main-p051"),
    ]
    neighbor = results[1]
    assert neighbor["distance"] is None
    assert neighbor["content"] and neighbor["page"] == 53 and neighbor["lecture"] == 2


def test_neighbors_stay_inside_the_same_document(monkeypatch):
    # 第2回の先頭ページの前に、第1回の最終ページを足さない
    monkeypatch.setattr(retrieval, "search", lambda query: _hits("lec02-main-p001"))
    results = retrieval.search_with_neighbors("第2回", window=1)
    assert _summary(results) == [
        ("lec02-main-p001", "hit", None),
        ("lec02-main-p002", "neighbor", "lec02-main-p001"),
    ]


def test_window_2_adds_two_pages_on_each_side(monkeypatch):
    monkeypatch.setattr(retrieval, "search", lambda query: _hits("lec03-main-p039"))
    results = retrieval.search_with_neighbors("メモリ", window=2)
    assert [r["page"] for r in results] == [39, 37, 38, 40, 41]


def test_tool_returns_neighbors_to_the_model(monkeypatch):
    monkeypatch.setattr(retrieval, "search", lambda query: _hits("lec03-main-p039"))
    # AIE_RAG_NEIGHBORS=1 で起動したときと同じ状態にする（既定のwindowを1に差し替える）
    monkeypatch.setattr(retrieval.search_with_neighbors, "__defaults__", (1,))
    model = TestModel(call_tools=["search_materials"])
    with agent.override(model=model):
        result = agent.run_sync("メモリの実装例は？")
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    ids = [r["page_id"] for r in json.loads(returns[0].content)["results"]]
    assert ids == ["lec03-main-p039", "lec03-main-p038", "lec03-main-p040"]
