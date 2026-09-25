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

## 評価結果 (未使用データ{n}件)

学習に一度も使っていない分割で評価している(JCommonsenseQA validation / chABSA test / JSNLI dev、
各タスク最大400件)。chABSAのtestは文単位の分割のため、同じ企業文書の別文が学習に含まれる可能性がある。

| 指標 | 値 |
| --- | --- |
| Accuracy | {accuracy} |
| ECE (Expected Calibration Error) | {ece} |
| Brier score | {brier} |
| NLL | {nll} |
{task_breakdown}
参考: 学習に使ったデータの一部を抜き出した評価では accuracy 0.938 / ECE 0.0114 と、未使用データより
高く出る。学習データと重なるため、未知データでの性能の目安には使えない。

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

### 評価結果 (per-channel動的int8量子化、学習データと重なる評価用サンプル{n}件)

未使用データでは測っていない。絶対値は過大だが、bf16(同じサンプルで0.970)との比較には使える。

| 指標 | 値 |
| --- | --- |
| Accuracy (overall) | {accuracy} |
| ECE (Expected Calibration Error) | {ece} |
| Brier score | {brier} |
| NLL | {nll} |
{task_breakdown}
フル精度(bf16、同じサンプルで0.970)より約7.5ポイント低い。同じサンプルで測ったGGUF版(0.955〜0.970)のほうが
精度が高いため、ローカル実行にはGGUF版(下記)を推奨する。
試行錯誤の詳細は `scripts/ask.py` の `load()` docstringと `qat.py` を参照。
"""


def _task_breakdown(eval_results: dict) -> str:
    task_names = sorted(
        key[: -len("_accuracy")] for key in eval_results if key.endswith("_accuracy") and key != "overall_accuracy"
    )
    if not task_names:
        return ""
    rows = "\n".join(
        f"| {task} | {eval_results[f'{task}_accuracy']:.4f} | {eval_results.get(f'{task}_n', 'N/A')} |" for task in task_names
    )
    return f"\n| タスク別 accuracy | 値 | n |\n| --- | --- | --- |\n{rows}\n"


def render_model_card(
    repo_id: str,
    eval_results: dict,
    base_model: str = BASE_MODEL_NAME,
    dataset_repo_id: str = "fukayatti0/jev-japanese-judgment",
) -> str:
    return MODEL_CARD_TEMPLATE.format(
        repo_id=repo_id,
        base_model=base_model,
        base_model_license_url=f"https://huggingface.co/{base_model}/blob/main/LICENSE",
        dataset_repo_id=dataset_repo_id,
        github_repo=GITHUB_REPO,
        n=eval_results.get("n", "N/A"),
        accuracy=_fmt(eval_results.get("overall_accuracy")),
        ece=_fmt(eval_results.get("ece")),
        brier=_fmt(eval_results.get("brier")),
        nll=_fmt(eval_results.get("nll")),
        task_breakdown=_task_breakdown(eval_results),
    )


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else "N/A" if value is None else str(value)


def render_quantized_section(repo_id: str, eval_results: dict) -> str:
    return "\n" + QUANTIZED_SECTION_TEMPLATE.format(
        repo_id=repo_id,
        n=eval_results.get("n", "N/A"),
        accuracy=_fmt(eval_results.get("overall_accuracy")),
        ece=_fmt(eval_results.get("ece")),
        brier=_fmt(eval_results.get("brier")),
        nll=_fmt(eval_results.get("nll")),
        task_breakdown=_task_breakdown(eval_results),
    )


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

    card = render_model_card(repo_id, eval_results or {}, base_model, dataset_repo_id)
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

    marker = "\n## 量子化版(ローカルPC/CPU向け)\n"
    card = card.split(marker)[0] + marker + render_quantized_section(repo_id, eval_results)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
