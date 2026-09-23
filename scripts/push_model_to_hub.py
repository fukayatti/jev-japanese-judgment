"""学習済みJevモデル(LoRAアダプタ+自作ヘッド)をHugging Face Hubに公開する。

ベースモデル(LFM2.5-1.2B-JP)の重みそのものは含めない(LFM Open License v1.0で
配布されているHF上のオリジナルからダウンロードされる想定)。ここで公開するのは
学習対象だった差分(LoRAアダプタ+自作ヘッド)だけ。
"""

from pathlib import Path

from huggingface_hub import HfApi

from model.head import BASE_MODEL_NAME
from scripts.push_to_hub import _get_hf_token

GITHUB_REPO = "https://github.com/fukayatti/jev-japanese-judgment"

MODEL_CARD_TEMPLATE = """---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/{base_model}/blob/main/LICENSE
base_model: {base_model}
language:
- ja
tags:
- jev
- judgment
- classification
- lora
- peft
---

# {repo_id}

{base_model} をバックボーンに、文脈と候補群から型付きの判定＋確率を直接返す
「Jev」スタイルの判定モデル。自己回帰的な文章生成の代わりに、`lm_head` を
候補ごとにスコアリングするカスタムヘッド(cross-encoder型)に差し替えている。

## アーキテクチャ

- バックボーン: {base_model} (凍結、LoRAで適応)
- ヘッド: 各候補を `[context][question][candidate_i]` としてエンコードし、
  最終トークンのhidden stateをMLPでスコア化、候補間でsoftmaxを取って
  確率分布を直接出力する(テキスト生成なし)

## 使い方

標準の `transformers` アーキテクチャではないため、`AutoModelForCausalLM` 等
では読み込めない。以下のリポジトリのコードを使うこと:

{github_repo}

```python
from eval.run_eval import load_model_from_checkpoint, predict
from data.convert.schema import JevExample

model = load_model_from_checkpoint("<このリポジトリをダウンロードしたパス>", "")
result = predict(model, [
    JevExample(
        id="example",
        source_dataset="manual",
        context="",
        question="日本の首都はどこ？",
        candidates=["大阪", "東京", "京都", "名古屋"],
        label=1,
        task_type="commonsense_qa",
    )
])
print(result)
```

## 学習データ

{dataset_repo_id}
(JCommonsenseQA / chABSA-dataset / JSNLI をJev形式に変換し、LLM拡張・
source_dataset間のリバランスを行ったもの)

## 評価結果 (held-out {n}件)

| 指標 | 値 |
| --- | --- |
| Accuracy | {accuracy} |
| ECE (Expected Calibration Error) | {ece} |
| Brier score | {brier} |
| NLL | {nll} |

## ライセンス

- コード: MIT ({github_repo})
- ベースモデル: [LFM Open License v1.0]({base_model_license_url}) ({base_model} より)

本リポジトリはLoRAアダプタと追加ヘッドの差分のみを配布しており、
ベースモデルの重みそのものは含まない。
"""


def push_model(
    checkpoint_dir: str,
    tag: str,
    repo_id: str,
    base_model: str = BASE_MODEL_NAME,
    dataset_repo_id: str = "fukayatti0/jev-japanese-judgment",
    eval_results: dict | None = None,
    private: bool = False,
) -> None:
    token = _get_hf_token()
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    checkpoint_path = Path(checkpoint_dir) / tag
    api.upload_folder(folder_path=str(checkpoint_path), repo_id=repo_id, repo_type="model")

    eval_results = eval_results or {}
    card = MODEL_CARD_TEMPLATE.format(
        repo_id=repo_id,
        base_model=base_model,
        base_model_license_url=f"https://huggingface.co/{base_model}/blob/main/LICENSE",
        dataset_repo_id=dataset_repo_id,
        github_repo=GITHUB_REPO,
        n=eval_results.get("n", "N/A"),
        accuracy=eval_results.get("overall_accuracy", "N/A"),
        ece=eval_results.get("ece", "N/A"),
        brier=eval_results.get("brier", "N/A"),
        nll=eval_results.get("nll", "N/A"),
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
