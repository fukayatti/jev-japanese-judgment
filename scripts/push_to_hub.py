"""変換済みJevデータセットをHugging Face Hubに公開する。"""

import pandas as pd
from datasets import Dataset

from data.convert.schema import JevExample

DATASET_LICENSE = "cc-by-sa-4.0"
DATASET_CITATION = (
    "JCommonsenseQA (JGLUE, CC BY-SA 4.0), "
    "chABSA-dataset (TIS, CC BY 4.0), "
    "JSNLI (Kyoto University, CC BY-SA 4.0)"
)


def to_dataset(examples: list[JevExample]) -> Dataset:
    df = pd.DataFrame([ex.model_dump() for ex in examples])
    return Dataset.from_pandas(df)


def push(examples: list[JevExample], repo_id: str, token: str, private: bool = False) -> None:
    dataset = to_dataset(examples)
    dataset.push_to_hub(repo_id, token=token, private=private)
