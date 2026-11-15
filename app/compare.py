"""複数のモデルと質問で、応答時間とトークン数を比較する。

使い方:
    uv run python -m app.compare

gemini-3.5-flash-lite と gemini-3.8-flash に、短い一般質問と講座固有の質問を1回ずつ送る。
呼び出しの間隔は15秒。429 と 503 は記録して、次の組み合わせへ進む。
"""

import time

from google import genai

from app.ask import DEFAULT_QUESTION, AskMeasurement, format_token_count, measure_question
from app.config import get_api_key

MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
)
QUESTIONS = (
    ("LoRA", "LoRAとは何ですか？"),
    ("宿題の3レベル", DEFAULT_QUESTION),
)
INTERVAL_SECONDS = 15.0


def run_comparison(
    client: genai.Client,
    models: tuple[str, ...],
    questions: tuple[tuple[str, str], ...],
    interval_s: float,
    sleep=time.sleep,
) -> list[AskMeasurement]:
    """モデルと質問の組み合わせを順に測る。最初の呼び出しの前には待たない。"""
    results: list[AskMeasurement] = []
    call_index = 0
    for model in models:
        for label, question in questions:
            if call_index > 0 and interval_s > 0:
                sleep(interval_s)
            results.append(measure_question(question, model, client, label))
            call_index += 1
    return results


def format_table(results: list[AskMeasurement]) -> str:
    """比較結果を Markdown の表にする。"""
    lines = [
        "| モデル | 質問 | 経過秒 | 入力トークン | 出力トークン | 思考トークン | 状態 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| {model} | {label} | {elapsed:.2f} | {prompt} | {output} | {thoughts} | {status} |".format(
                model=result.model,
                label=result.question_label,
                elapsed=result.elapsed_s,
                prompt=format_token_count(result.prompt_tokens),
                output=format_token_count(result.output_tokens),
                thoughts=format_token_count(result.thoughts_tokens),
                status=result.status,
            )
        )
    return "\n".join(lines)


def _sleep_and_notify(seconds: float) -> None:
    print(f"[wait] {seconds:.0f}秒")
    time.sleep(seconds)


def main() -> None:
    print(f"[interval] {INTERVAL_SECONDS:.0f}秒")
    for model in MODELS:
        print(f"[model] {model}")
    for label, question in QUESTIONS:
        print(f"[質問] {label}: {question}")
    print()

    client = genai.Client(api_key=get_api_key())
    results = run_comparison(
        client,
        MODELS,
        QUESTIONS,
        INTERVAL_SECONDS,
        sleep=_sleep_and_notify,
    )
    print()
    print(format_table(results))


if __name__ == "__main__":
    main()
