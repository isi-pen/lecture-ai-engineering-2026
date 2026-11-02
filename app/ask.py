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
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


@dataclass(frozen=True)
class AskMeasurement:
    """1回の生成呼び出しの計測結果。"""

    model: str
    question_label: str
    question: str
    text: str | None
    elapsed_s: float
    prompt_tokens: int | None
    output_tokens: int | None
    thoughts_tokens: int | None
    status: str
    error_detail: str = ""


def format_token_count(value: int | None) -> str:
    """トークン数が無いとき unknown にする。"""
    if value is None:
        return "unknown"
    return str(value)


def measure_question(
    question: str,
    model: str,
    client: genai.Client,
    question_label: str,
) -> AskMeasurement:
    """1回呼び出し、経過時間とトークン数を返す。APIエラーは結果の status に残す。"""
    started = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=question)
    except errors.APIError as exc:
        return AskMeasurement(
            model=model,
            question_label=question_label,
            question=question,
            text=None,
            elapsed_s=time.perf_counter() - started,
            prompt_tokens=None,
            output_tokens=None,
            thoughts_tokens=None,
            status=str(exc.code),
            error_detail=exc.message or "",
        )

    usage = response.usage_metadata
    return AskMeasurement(
        model=model,
        question_label=question_label,
        question=question,
        text=response.text,
        elapsed_s=time.perf_counter() - started,
        prompt_tokens=_token_count(usage, "prompt_token_count"),
        output_tokens=_token_count(usage, "candidates_token_count"),
        thoughts_tokens=_token_count(usage, "thoughts_token_count"),
        status="ok",
    )


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    result = measure_question(question, MODEL, client, question_label="")
    if result.status != "ok":
        raise errors.APIError(int(result.status), {"message": result.error_detail})

    print(f"[time] {result.elapsed_s:.2f}秒")
    print(_token_line(result))
    return result.text


def _token_count(usage: object, name: str) -> int | None:
    value = getattr(usage, name, None)
    if value is None:
        return None
    return int(value)


def _token_line(result: AskMeasurement) -> str:
    line = (
        f"[tokens] 入力: {format_token_count(result.prompt_tokens)}, "
        f"出力: {format_token_count(result.output_tokens)}"
    )
    if result.thoughts_tokens is not None:
        line += f", 思考: {result.thoughts_tokens}"
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
