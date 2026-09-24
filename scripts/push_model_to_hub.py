"""学習済みJevモデル(LoRAアダプタ+自作ヘッド)をHugging Face Hubに公開する。

ベースモデル(LFM2.5-1.2B-JP)の重みそのものは含めない(LFM Open License v1.0で
配布されているHF上のオリジナルからダウンロードされる想定)。ここで公開するのは
学習対象だった差分(LoRAアダプタ+自作ヘッド)だけ。
"""

import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

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


QUANTIZED_SECTION_TEMPLATE = """GPU無しのローカルPC向けに、per-channel動的int8量子化をかけたモデルを
`quantized/model.pt` として同梱している。LoRAはバックボーンへマージ済みで、
量子化された重みを含むモデル全体を `torch.save` でpickle化したもの
(量子化Linear層のパックされたパラメータはstate_dictでの再構築が煩雑なため)。

```python
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download("{repo_id}", "quantized/model.pt")
model = torch.load(path, weights_only=False, map_location="cpu")
model.eval()
```

もしくは、Hubからダウンロード後にその場で量子化する方法でも同じ結果になる
(起動のたびに量子化する分だけ少し遅い):

```
python -m scripts.ask --device cpu --quantize --question "..." --candidates "..."
```

### 評価結果 (held-out {n}件、per-channel動的int8量子化時)

| 指標 | 値 |
| --- | --- |
| Accuracy (overall) | {accuracy} |
| ECE (Expected Calibration Error) | {ece} |
| Brier score | {brier} |
| NLL | {nll} |
{task_breakdown}
フル精度(bf16)と比べて数%の精度劣化があり、特にNLIタスクで大きい。
試行錯誤の詳細は `scripts/ask.py` の `load()` docstringと `qat.py` を参照。
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


def push_quantized_model(
    repo_id: str,
    quantized_eval_results: dict | None = None,
) -> None:
    """公開済みのフル精度モデル(repo_id)をper-channel動的int8量子化し、
    同じリポジトリの quantized/model.pt として追加する。

    量子化はCPU向けバックエンド(ONEDNN)前提のため、常にdevice="cpu"で
    scripts.ask.load(..., quantize=True)を呼ぶ(GPU上でのquantize_dynamic
    呼び出しは想定していない)。state_dictではなくtorch.save(model, ...)で
    モデルオブジェクト全体をpickle化する
    (量子化Linear層のパックされたパラメータはstate_dictの再構築が煩雑なため。
    ロード側はtorch.load(path, weights_only=False)を使う想定)。
    """
    from scripts.ask import load as load_ask_model

    token = _get_hf_token()
    api = HfApi(token=token)

    model = load_ask_model(repo_id, device="cpu", quantize=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir) / "model.pt"
        torch.save(model, local_path)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo="quantized/model.pt",
            repo_id=repo_id,
            repo_type="model",
        )

    _append_quantized_section_to_card(api, repo_id, quantized_eval_results or {})


def _append_quantized_section_to_card(api: HfApi, repo_id: str, eval_results: dict) -> None:
    """既存のREADME.mdに量子化版の説明セクションを追記する(全体は上書きしない)。
    再実行時に重複しないよう、既にセクションがあれば一旦切り落としてから追記し直す。
    """
    readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="model", token=api.token)
    card = Path(readme_path).read_text(encoding="utf-8")

    task_names = sorted(
        key[: -len("_accuracy")] for key in eval_results if key.endswith("_accuracy") and key != "overall_accuracy"
    )
    task_breakdown = ""
    if task_names:
        rows = "\n".join(
            f"| {task} | {eval_results[f'{task}_accuracy']} | {eval_results.get(f'{task}_n', 'N/A')} |"
            for task in task_names
        )
        task_breakdown = f"\n| タスク別 accuracy | 値 | n |\n| --- | --- | --- |\n{rows}\n"

    marker = "\n## 量子化版(ローカルPC/CPU向け)\n"
    card = card.split(marker)[0]
    card += marker + QUANTIZED_SECTION_TEMPLATE.format(
        repo_id=repo_id,
        n=eval_results.get("n", "N/A"),
        accuracy=eval_results.get("overall_accuracy", "N/A"),
        ece=eval_results.get("ece", "N/A"),
        brier=eval_results.get("brier", "N/A"),
        nll=eval_results.get("nll", "N/A"),
        task_breakdown=task_breakdown,
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
