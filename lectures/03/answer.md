# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：
第１回は一般論のような回答で、基礎・実装・発展という言葉は出てこなかった。
今回は基礎・実装・発展について、用語集どおりの回答が出てきた。

- `--trace` で見えたツール呼び出し（ツール名と引数）：
[tool call]   search_glossary({'query': '3レベル'})

- 1つの質問での LLM 呼び出し回数（`requests`）：
[usage] RunUsage(cost=Decimal('0.0005349'), details={'text_prompt_tokens': 783}, input_text_tokens=783, input_tokens=783, output_tokens=120, requests=2, tool_calls=1)

- 追加したツールと、docstring に書いた説明：
def make_quiz(lecture: int) -> str:
    """指定した回の用語から１つを選び、
    確認問題と答えを返す。
    
    「第２回の確認問題を出して」のように、
    復習用の問題を求められたときに使う。
    """
    # 1.load_glossary()からlectureが一致する用語を集める
    # 2.1つ選ぶ(random.choice)
    # 3.問題と答えを文字列で返す(json.dumpsなど)

    entries = [e for e in load_glossary() if e["lecture"] == lecture]
    if not entries:
        return f"第{lecture}回の用語は見つかりませんでした。"
    entry = random.choice(entries)
    return json.dumps(
        {
            "question": f"「{entry['term']}」の意味は？",
            "answer": entry["definition"],
        },
        ensure_ascii=False,
    )

## 宿題

- 取り組んだレベル：基礎 / 実装 / 発展
基礎

- やったこと：
INSTRUCTIONSの
- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。
- 回の内容を聞かれたら get_lecture_summary を使う。
を次のように変更した。
- 用語の違いを聞かれたら search_glossary は使わない。
- 必ず get_lecture_summary を1回だけ呼び、要点に書いてある違いだけを答える。
- 回答は「違い:」で始まる1文にする。最後に回の番号は書かない。

search_glossaryのdocstringに次を追加した。
    用語の違いを聞かれたときは、このツールを使わない。
    その場合は get_lecture_summary を使う。   

- 結果：
次のような出力となった。

違い: LoRAが事前学習済みモデルの重みを固定して低ランク行列を追加し効率的にファインチューニングを行う手法であるのに対し、QLoRAはその重みを4量子化してメモリ使用量をさらに削減する手法。

--- trace ---
[tool call]   get_lecture_summary({'lecture': 1})
[tool return] get_lecture_summary -> # 第1回 AIエンジニアリング概論と開発環境

## この回のねらい

- AIエンジニアリングが「モデルを作る」ことより「モデルを部品として使い、評価と改善を回す」ことに重心がある領域だと理解する
- 講座を通して使う開発環境（GitH…


- 気づき・疑問：
INSTRUCTIONS を変えると、search_glossaryではなくget_lecture_summary が呼ばれるようになった。
INSTRUCTIONSでツールを変更できたが、LoRA, QLoRAは第2回の内容なので、
    [tool call]   get_lecture_summary({'lecture': 2})
となるべきが、
    [tool call]   get_lecture_summary({'lecture': 1})
となっていた。
ツールで取得できなかった答えをモデル知識で補っているようです。
