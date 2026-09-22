"""変換済みJevデータセットをHugging Face Hubに公開する。

認証は環境変数 HF_TOKEN から取得する(Colab/ローカルどちらも `os.environ["HF_TOKEN"]` を
事前に設定しておくこと。Colabならユーザーシークレット経由で環境変数に注入する)。
"""

import os

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


def _get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "環境変数 HF_TOKEN が設定されていません。"
            "Colabならユーザーシークレットで登録するか、`os.environ['HF_TOKEN'] = ...` で設定してください。"
        )
    return token


def push(examples: list[JevExample], repo_id: str, private: bool = False) -> None:
    dataset = to_dataset(examples)
    dataset.push_to_hub(repo_id, token=_get_hf_token(), private=private)
