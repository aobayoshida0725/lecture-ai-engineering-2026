"""第3回 宿題（実装）: 承認付きツール save_learning_log と、承認待ち → 確認 → 再開のテスト。

実モデル・API キー・課金は使わない。
- 外部モデルへのリクエストは models.ALLOW_MODEL_REQUESTS = False で禁止する
- モデルの代わりに FunctionModel（自分で書いた関数が応答する偽モデル）と TestModel を使う
- 保存先 tools.LOG_DIR は tmp_path に差し替える（実際の learning_log/ には何も残さない）
"""

import builtins
import sys
from datetime import datetime

import pytest
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolDenied, models
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

import app.agent as cli  # main() などの CLI も呼ぶので、モジュールごと読み込む
from app import tools

QUESTION = "今日の学びを学習記録に保存して"
TITLE = "第3回 エージェントの仕組み"
CONTENT = "エージェントはモデル＋ループ＋ツール。\n保存の前に承認を挟む方法も学んだ。"
ANSWER = "学習記録の件、対応しました。"


def save_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """偽モデル: 1回目は save_learning_log を呼び、ツールの結果が戻ってきたら文章で答える。"""
    if len(messages) == 1:
        args = {"title": TITLE, "content": CONTENT}
        return ModelResponse(parts=[ToolCallPart("save_learning_log", args)])
    return ModelResponse(parts=[TextPart(ANSWER)])


def answer_only(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """偽モデル: ツールを呼ばずに答える。"""
    return ModelResponse(parts=[TextPart("LoRAは少ないパラメータで調整する方法です。")])


@pytest.fixture(autouse=True)
def log_dir(monkeypatch, tmp_path):
    """全テスト共通: 実モデルへのリクエストを禁止し、保存先を tmp_path に差し替える。"""
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # 万一 build_model() が呼ばれても API には届かない
    path = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LOG_DIR", path)
    return path


def saved_files(log_dir):
    return sorted(log_dir.glob("*.md")) if log_dir.exists() else []


def tool_returns(result, name):
    return [
        p for m in result.all_messages() for p in m.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == name
    ]


# --- ツール単体 ---------------------------------------------------------------


def test_save_learning_log_writes_markdown(log_dir):
    message = tools.save_learning_log(TITLE, CONTENT)
    [path] = saved_files(log_dir)
    assert path.name in message
    text = path.read_text(encoding="utf-8")
    assert text.startswith(f"# {TITLE}\n")
    assert CONTENT in text


def test_save_learning_log_does_not_overwrite(monkeypatch, log_dir):
    class FixedDatetime:  # 2回とも同じ時刻にして、同じファイル名になる状況を作る
        @staticmethod
        def now():
            return datetime(2026, 11, 9, 10, 0, 0)

    monkeypatch.setattr(tools, "datetime", FixedDatetime)
    tools.save_learning_log("1回目", "最初の記録")
    tools.save_learning_log("2回目", "次の記録")
    assert [p.name for p in saved_files(log_dir)] == ["2026-11-09_100000.md", "2026-11-09_100000_2.md"]
    assert "最初の記録" in (log_dir / "2026-11-09_100000.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_save_learning_log_skips_empty_record(log_dir, content):
    assert "保存しませんでした" in tools.save_learning_log(TITLE, content)
    assert not log_dir.exists()  # フォルダも作らない


# --- エージェント: 承認待ち → 再開 ----------------------------------------------


def run_until_approval():
    """FunctionModel でエージェントを動かす（save_learning_log を呼ぶので、承認待ちで止まる）。"""
    with cli.agent.override(model=FunctionModel(save_then_answer)):
        return cli.agent.run_sync(QUESTION)


def resume(first, decision):
    """承認待ちの run を、確認の結果 decision（True / ToolDenied）を渡して再開する。"""
    [call] = first.output.approvals
    with cli.agent.override(model=FunctionModel(save_then_answer)):
        return cli.agent.run_sync(
            message_history=first.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: decision}),
        )


def test_agent_waits_for_approval_without_creating_files(log_dir):
    result = run_until_approval()
    assert isinstance(result.output, DeferredToolRequests)
    [call] = result.output.approvals
    assert call.tool_name == "save_learning_log"
    assert not log_dir.exists()  # 承認前は、フォルダもファイルも作られていない


def test_testmodel_call_also_waits_for_approval(log_dir):
    with cli.agent.override(model=TestModel(call_tools=["save_learning_log"])):
        result = cli.agent.run_sync(QUESTION)
    assert isinstance(result.output, DeferredToolRequests)
    assert not log_dir.exists()


def test_approved_call_is_saved_after_resume(log_dir):
    result = resume(run_until_approval(), True)
    assert result.output == ANSWER
    [path] = saved_files(log_dir)
    assert CONTENT in path.read_text(encoding="utf-8")


def test_denied_call_is_not_saved(log_dir):
    result = resume(run_until_approval(), ToolDenied("保存しないでください"))
    assert result.output == ANSWER
    assert not log_dir.exists()
    # 拒否したことは、ツールの結果としてモデルに伝わっている
    assert "保存しないでください" in str(tool_returns(result, "save_learning_log")[0].content)


# --- CLI: 表示 → 確認 → 再開 ----------------------------------------------------


def run_cli(monkeypatch, argv, answer):
    """CLI（main）を偽モデルで実行する。answer は input() への返事。例外クラスなら、それを送出する。"""
    monkeypatch.setattr(sys, "argv", ["app.agent", *argv])
    monkeypatch.setattr(cli, "build_model", lambda: FunctionModel(save_then_answer))

    def fake_input(prompt=""):
        print(prompt)
        if isinstance(answer, type) and issubclass(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(builtins, "input", fake_input)
    cli.main()


@pytest.mark.parametrize("answer", ["y", "yes", " YES ", "はい"])
def test_cli_saves_only_after_explicit_yes(monkeypatch, capsys, log_dir, answer):
    run_cli(monkeypatch, [QUESTION], answer)
    out = capsys.readouterr().out
    assert "[承認待ち] save_learning_log" in out
    assert f"title: {TITLE}" in out  # 保存予定の内容を、確認の前に表示している
    assert "保存の前に承認を挟む方法も学んだ。" in out
    assert len(saved_files(log_dir)) == 1
    assert ANSWER in out  # 再開して、最後の回答まで進んでいる


@pytest.mark.parametrize("answer", ["", "n", "no", "いいえ", "yy", EOFError, KeyboardInterrupt])
def test_cli_does_not_save_without_explicit_yes(monkeypatch, capsys, log_dir, answer):
    run_cli(monkeypatch, [QUESTION], answer)
    out = capsys.readouterr().out
    assert "[承認待ち] save_learning_log" in out
    assert not log_dir.exists()
    assert ANSWER in out  # 拒否しても、会話は再開して回答まで進む


def test_cli_trace_shows_tool_calls_after_resume(monkeypatch, capsys, log_dir):
    run_cli(monkeypatch, ["--trace", QUESTION], "y")
    out = capsys.readouterr().out
    assert "--- trace ---" in out
    assert "[tool call]   save_learning_log" in out
    assert "[tool return] save_learning_log -> 学習記録を learning_log/" in out
    assert "[usage]" in out


def test_resume_keeps_history_and_usage(monkeypatch, log_dir):
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    result = cli.run_with_approval(cli.agent, QUESTION, FunctionModel(save_then_answer))
    assert result.output == ANSWER
    first_request = result.all_messages()[0]
    assert any(isinstance(p, UserPromptPart) and p.content == QUESTION for p in first_request.parts)
    assert result.usage.requests == 2  # 承認前の1回 + 再開後の1回（usage を引き継いでいる）


def test_cli_list_tools_shows_all_tools(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["app.agent", "--list-tools"])
    cli.main()
    out = capsys.readouterr().out
    for name in ["search_glossary", "get_lecture_summary", "make_quiz", "save_learning_log"]:
        assert f"[{name}]" in out


def test_cli_mcp_mode_answers_without_asking(monkeypatch, capsys, log_dir):
    """--mcp（承認付きツールが無いエージェント）では、確認せずにそのまま回答する。"""
    monkeypatch.setattr(sys, "argv", ["app.agent", "--mcp", "LoRAとは？"])
    monkeypatch.setattr(cli, "build_mcp_agent", lambda: Agent(instructions=cli.INSTRUCTIONS))
    monkeypatch.setattr(cli, "build_model", lambda: FunctionModel(answer_only))
    monkeypatch.setattr(builtins, "input", lambda prompt="": pytest.fail("確認は不要のはず"))
    cli.main()
    out = capsys.readouterr().out
    assert "LoRAは少ないパラメータで調整する方法です。" in out
    assert not log_dir.exists()
