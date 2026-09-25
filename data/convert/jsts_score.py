"""JSTS(JGLUE) -> Jev形式の段階評価(score)への決定論的変換。

元データ: zenless-lab/jsts (HF, parquet、元はyahoojapan/JGLUE、CC BY-SA 4.0)
  sentence1, sentence2, label(0.0〜5.0の類似度、複数人の評価の平均)
  分割: train / validation

類似度を最も近い整数の段階(0〜5)に丸めて、段階の説明文(ルーブリック)から1つ選ぶ形にする。
説明文はSemEvalのSTSの定義に沿った日本語。学習はtrain、評価はvalidationを使う。
"""

from data.convert.schema import JevExample

SOURCE_DATASET = "zenless-lab/jsts"

STS_LEVELS_JA = [
    "0: まったく異なる内容で、関係がない",
    "1: 同じ話題だが、意味は等しくない",
    "2: 意味は等しくないが、いくつかの詳細を共有している",
    "3: おおむね同じ意味だが、重要な情報が異なる",
    "4: ほぼ同じ意味で、細部だけが異なる",
    "5: 完全に同じ意味",
]
QUESTION_JA = "2つの文の意味の近さは、次のどの段階に最も近いですか？"


def level_of(score: float) -> int:
    return min(5, max(0, int(score + 0.5)))


def row_to_example(row: dict, split: str, index: int) -> JevExample:
    return JevExample(
        id=f"jsts_{split}_{row.get('sentence_pair_id', index)}",
        source_dataset="jsts",
        context=f"文1: {row['sentence1'].strip()}\n文2: {row['sentence2'].strip()}",
        question=QUESTION_JA,
        candidates=list(STS_LEVELS_JA),
        label=level_of(float(row["label"])),
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
