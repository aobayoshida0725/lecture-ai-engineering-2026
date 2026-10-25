"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, chat) -> str:
    response = chat.send_message(question)

    usage = response.usage_metadata
    if usage is not None:
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        print(f"[入力トークン数] {input_tokens if input_tokens is not None else '不明'}")
        print(f"[出力トークン数] {output_tokens if output_tokens is not None else '不明'}")
    else:
        print("[トークン数] 使用量の情報を取得できませんでした")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="履歴を保ってLLMと対話する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    #チャットの作成をループ外で1回行う
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)

    print(f"[model] {MODEL}")
    print("終了するには /exit または /quit を入力してください。")
    question = args.question.strip()

    try:
        while True:
            if question.lower() in {"/exit", "/quit"}:
                break
            if question:
                print(f"\n[質問] {question}")
                answer = ask(question, chat)
                print(f"[回答] \n{answer}")
            question = input("\n次の質問 > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n対話を終了します。")
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
