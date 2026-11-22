"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
import random
from datetime import datetime
from functools import cache
from pathlib import Path

from app.config import api_error_message

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# 第3回 宿題（実装）: 学習記録の保存先（.gitignore 済み）。テストでは tmp_path に差し替える
LOG_DIR = Path(__file__).resolve().parent.parent / "learning_log"
MAX_RESULTS = 3


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> str:
    """指定した回の用語から復習クイズを1問作る。

    ユーザーが「第N回のクイズを出して」など、復習用の問題を求めたときに呼ぶ。
    用語集からその回の用語を1つランダムに選び、問題と答えを返す。

    Args:
        lecture: 講義の回番号（1始まり。第1回なら 1）。

    Returns:
        question（問題）と answer（答え）を持つ JSON 文字列。
        その回の用語が無いときは、その旨を伝える日本語メッセージ。
    """
    # 1. 指定した回の用語だけを集める
    entries = [e for e in load_glossary() if e["lecture"] == lecture]

    # 2. 該当なし（0・負数・範囲外を含む）は例外にせずメッセージを返す
    if not entries:
        return f"第{lecture}回の用語は見つかりませんでした。用語集にある回（1始まり）を指定してください。"

    # 3. 1語選んで問題と答えを作り、日本語のまま JSON にする
    entry = random.choice(entries)
    quiz = {
        "question": f"「{entry['term']}」の意味は？",
        "answer": entry["definition"],
    }
    return json.dumps(quiz, ensure_ascii=False)


def save_learning_log(title: str, content: str) -> str:
    """学習の記録を learning_log/ フォルダに Markdown ファイルとして保存する。

    利用者が「今日学んだことを記録して」「学習ログに残して」などと頼んだときに使う。
    保存は取り消しにくい操作なので、実行の前に利用者の承認が必要
    （agent.py で requires_approval=True を付けて登録している）。

    Args:
        title: 記録の見出し（例: "第3回 エージェントの仕組み"）。
        content: 記録の本文（学んだこと、疑問に思ったことなど）。

    Returns:
        保存したファイル名、または保存しなかった理由。
    """
    if not content.strip():
        return "記録の本文が空なので、保存しませんでした。"

    # ファイル名は日時だけで作る（引数を使わないので、learning_log/ の外には書けない）
    now = datetime.now()
    stem = now.strftime("%Y-%m-%d_%H%M%S")
    heading = title.strip() or "学習記録"
    text = f"# {heading}\n\n- 保存日時: {now:%Y-%m-%d %H:%M:%S}\n\n{content.strip()}\n"

    LOG_DIR.mkdir(exist_ok=True)
    n = 1
    while True:
        name = f"{stem}.md" if n == 1 else f"{stem}_{n}.md"
        try:
            # "x" は新規作成専用。同じ名前のファイルがあると FileExistsError になり、上書きしない
            with open(LOG_DIR / name, "x", encoding="utf-8") as f:
                f.write(text)
        except FileExistsError:
            n += 1
            continue
        return f"学習記録を learning_log/{name} に保存しました。"


def search_materials(query: str) -> str:
    """第1〜3回の本講義資料を検索し、関連ページの本文・タイトル・ページ番号を返す。

    queryに調べたい内容を指定する。
    """
    from google.genai.errors import APIError
    import sqlite3
    # 第5回 宿題（実装）: AIE_RAG_NEIGHBORS が1以上なら前後ページも返す。0なら通常の検索と同じ。
    # 比較の条件をそろえるため、このdocstring（モデルへの説明文）は変えていない
    from app.retrieval import search_with_neighbors

    try:
        return json.dumps({"results": search_with_neighbors(query)}, ensure_ascii=False)
    except (ValueError, FileNotFoundError, sqlite3.Error) as e:
        return json.dumps({"error": str(e), "results": []}, ensure_ascii=False)
    except APIError as e:
        return json.dumps({"error": api_error_message(e.code), "results": []}, ensure_ascii=False)


def browse_tree(node_id: str | None = None) -> str:
    """目次とページの先頭文を返す。ページIDを指定すると全文を返す。

    最初はnode_idを省略する。続いて返されたIDを指定して目次をたどる。
    """
    from app.retrieval import browse_tree as browse

    try:
        return json.dumps(browse(node_id), ensure_ascii=False)
    except (ValueError, OSError) as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def search_graph(query: str) -> str:
    """資料を検索し、前後ページ・共通用語をたどって関連ページの本文と引用情報を返す。

    queryに調べたい内容を指定する。
    """
    import sqlite3
    from google.genai.errors import APIError
    from app.retrieval import search_graph as search

    try:
        return json.dumps(search(query), ensure_ascii=False)
    except (ValueError, OSError, sqlite3.Error) as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except APIError as e:
        return json.dumps({"error": api_error_message(e.code)}, ensure_ascii=False)
