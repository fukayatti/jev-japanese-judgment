"""公開済みのJevモデルに実際に質問を投げて試すための単体スクリプト。

Colabのチェックポイント(Drive)には依存せず、Hugging Face Hubから
直接モデルをダウンロードして使う。

使い方 (CLI):
  python -m scripts.ask --question "日本の首都はどこ？" --candidates "大阪,東京,京都,名古屋"

使い方 (Python):
  from scripts.ask import load, ask
  model = load()
  ask(model, "日本の首都はどこ？", ["大阪", "東京", "京都", "名古屋"])
"""

import argparse

import torch

from data.convert.schema import JevExample
from eval.run_eval import load_model_from_hub, predict

DEFAULT_REPO_ID = "fukayatti0/jev-japanese-judgment"


def load(repo_id: str = DEFAULT_REPO_ID, device: str = "cuda", quantize: bool = False):
    """quantize=Trueで動的int8量子化をかける。GPU無しのローカルPCでの実行を想定した
    軽量化オプションで、CPU実行時のみ意味がある(量子化バックエンドがCPU向けのため)。
    PEFTのLoRA層はそのままだとtorch.nn.Linearと型が一致せず量子化対象から漏れることが
    あるので、先にLoRAをバックボーンへマージしてから量子化する
    (推論専用にする前提。以後この重みで追加学習はできなくなる)。

    量子化バックエンド(CPU向けONEDNN)はfloat32の入力しか受け付けないため
    ("qlinear_dynamic (ONEDNN): data type of input should be float" で実際に
    エラーになった)、モデルの重み自体をbfloat16からfloat32へキャストしてから
    量子化する。

    以下、精度回復を試した経緯(held-out 200件でoverall accuracy):
      - bf16フル精度: 97.0%
      - 全層をdefault_dynamic_qconfig(per-tensor)で量子化: 89.0% (nli 79.7%)
      - self_attnを除きper-tensorで量子化(選択的PTQ): 86.5% (悪化、仮説外れ)
      - 選択的PTQ層をQAT(fake quant込みで300ステップ再学習): 78.5%
        (9.7億パラメータをLoRA無しでフル更新、300ステップ/0.09epoch分の
        データだけでは適応というより既存の重みを壊した可能性が高い)
    選択的量子化・QATはどちらも悪化したため撤回。代わりに、量子化そのものの
    精度(per-tensor: 重み行列全体で1つのスケール値)を、per-channel
    (出力チャネルごとに別のスケール値、再学習不要で精度が上がりやすい
    定番のテクニック)に変えて全層量子化を試す。
    """
    if quantize and device != "cpu":
        raise ValueError("動的量子化はCPU向けなので device='cpu' と併用すること")

    model = load_model_from_hub(repo_id, device=device)
    if quantize:
        model.backbone = model.backbone.merge_and_unload()
        model = model.float()
        qconfig_spec = {
            name: torch.ao.quantization.per_channel_dynamic_qconfig
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        model = torch.ao.quantization.quantize_dynamic(model, qconfig_spec, dtype=torch.qint8)
    return model


def ask(model, question: str, candidates: list[str], context: str = "", device: str = "cuda") -> dict:
    """1問だけ手軽に投げて結果を見るためのラッパー。labelは推論に使わないのでダミー値(0)でよい。"""
    example = JevExample(
        id="query",
        source_dataset="manual",
        context=context,
        question=question,
        candidates=candidates,
        label=0,
        task_type="manual",
    )
    return predict(model, [example], device=device)[0]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Jevモデルに質問を投げる")
    parser.add_argument("--question", required=True, help="質問文")
    parser.add_argument("--candidates", required=True, help="カンマ区切りの候補一覧")
    parser.add_argument("--context", default="", help="文脈(省略可)")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Faceのモデルリポジトリ")
    parser.add_argument("--device", default="cuda", help="cuda または cpu")
    parser.add_argument(
        "--quantize", action="store_true", help="動的int8量子化をかける(--device cpu と併用、GPU無しのPC向け)"
    )
    args = parser.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    model = load(args.repo_id, device=args.device, quantize=args.quantize)
    result = ask(model, args.question, candidates, context=args.context, device=args.device)

    print(f"question: {result['question']}")
    print(f"predicted: {result['predicted']}")
    for candidate, prob in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {candidate}: {prob:.4f}")


if __name__ == "__main__":
    _main()
