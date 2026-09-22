"""JCommonsenseQA (JGLUE) -> Jev形式の決定論的変換。

元データ: sbintuitions/JCommonsenseQA (Hugging Face, parquet, CC BY-SA 4.0)
  q_id: int
  question: str
  choice0..choice4: str
  label: int (0-4)

備考: shunk031/JGLUE はloading scriptベースで配布されており、
`datasets` ライブラリがdataset scriptsのサポートを終了したため読み込めなくなった
(RuntimeError: Dataset scripts are no longer supported)。
そのためparquet形式で配布されている sbintuitions/JCommonsenseQA を使う。
"""

from datasets import load_dataset

from data.convert.schema import JevExample

SOURCE_DATASET = "sbintuitions/JCommonsenseQA"


def convert(split: str = "train") -> list[JevExample]:
    ds = load_dataset(SOURCE_DATASET, split=split)
    examples = []
    for row in ds:
        candidates = [row[f"choice{i}"] for i in range(5)]
        examples.append(
            JevExample(
                id=f"jcommonsenseqa_{row['q_id']}",
                source_dataset="jcommonsenseqa",
                context="",
                question=row["question"],
                candidates=candidates,
                label=row["label"],
                task_type="commonsense_qa",
            )
        )
    return examples


if __name__ == "__main__":
    examples = convert("train")
    print(f"converted {len(examples)} examples")
    print(examples[0])
