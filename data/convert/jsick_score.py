"""JSICK -> Jev形式の段階評価(score)への決定論的変換。

元データ: mteb/JSICK (HF, parquet)。元はverypluming/JSICK(SICKの日本語版、人手翻訳+再アノテーション)。
  sentence1, sentence2, score(1.0〜5.0の関連度、複数人の評価の平均)
  分割: train / validation / test
ライセンス: mtebのカードはCC BY 4.0、元リポジトリはLICENSEがCC BY-SA 4.0でREADMEはCC BY 4.0と表記が
食い違う。どちらでも、CC BY-SA 4.0の本データセットに取り込める。

関連度を最も近い整数(1〜5)に丸め、段階の説明文から1つ選ぶ形にする。含意ラベルはmteb版に無いので使わない。
"""

from data.convert.schema import JevExample

SOURCE_DATASET = "mteb/JSICK"

RELATEDNESS_LEVELS_JA = [
    "1: まったく関連がない",
    "2: ほとんど関連がない",
    "3: いくらか関連がある",
    "4: かなり関連がある",
    "5: 非常に強く関連がある",
]
QUESTION_JA = "2つの文の関連の強さは、次のどの段階に最も近いですか？"


def level_of(score: float) -> int:
    """1.0〜5.0 -> 0〜4(候補のインデックス)"""
    return min(4, max(0, int(score + 0.5) - 1))


def row_to_example(row: dict, split: str, index: int) -> JevExample:
    return JevExample(
        id=f"jsick_{split}_{index}",
        source_dataset="jsick",
        context=f"文1: {row['sentence1'].strip()}\n文2: {row['sentence2'].strip()}",
        question=QUESTION_JA,
        candidates=list(RELATEDNESS_LEVELS_JA),
        label=level_of(float(row["score"])),
        task_type="score",
    )


def convert(split: str = "train", rows: list[dict] | None = None) -> list[JevExample]:
    if rows is None:
        from datasets import load_dataset

        rows = load_dataset(SOURCE_DATASET, split=split)
    return [row_to_example(row, split, i) for i, row in enumerate(rows)]


if __name__ == "__main__":
    examples = convert("train")
    print(f"converted {len(examples)} examples")
    print(examples[0])
