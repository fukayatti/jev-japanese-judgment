"""JNLI(JGLUE) -> Jev形式のyes/no(noul)への決定論的変換。

元データ: zenless-lab/jnli (HF, parquet, CC BY-SA 4.0、元はyahoojapan/JGLUE)
  premise, hypothesis, label: 0=entailment / 1=neutral / 2=contradiction (カードと実例で確認済み)
  "test"分割はJGLUEのvalidation(2,434件)。JGLUE本来のtestはラベル非公開。

3値のNLIを「hypothesisはこの文から確実に言えるか」のyes/noに落とす:
entailmentだけが「はい」、neutralとcontradictionは「いいえ」。
学習にはtrain、評価にはtest(=JGLUEのvalidation)を使い、混ぜないこと。
"""

from data.convert.schema import JevExample

SOURCE_DATASET = "zenless-lab/jnli"
_CANDIDATES_JA = ["はい", "いいえ"]


def row_to_example(row: dict, split: str, index: int) -> JevExample:
    return JevExample(
        id=f"jnli_{split}_{index}",
        source_dataset="jnli",
        context=row["premise"].strip(),
        question=f"「{row['hypothesis'].strip()}」は、この文から確実に言えますか？",
        candidates=list(_CANDIDATES_JA),
        label=0 if row["label"] == 0 else 1,
        task_type="noul",
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
