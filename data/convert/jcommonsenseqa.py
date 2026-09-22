"""JCommonsenseQA (JGLUE) -> Jev形式の決定論的変換。

元データ (shunk031/JGLUE, config="JCommonsenseQA"):
  q_id: str
  question: str
  choice0..choice4: str
  label: int (0-4)
"""

from datasets import load_dataset

from data.convert.schema import JevExample


def convert(split: str = "train") -> list[JevExample]:
    ds = load_dataset("shunk031/JGLUE", name="JCommonsenseQA", split=split)
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
