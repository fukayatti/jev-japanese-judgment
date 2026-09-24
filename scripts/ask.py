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

    全Linear層を一律量子化すると、held-out 200件でoverall accuracy 97.0%→89.0%、
    特にnliタスクが93.7%→79.7%と大きく劣化することを実測で確認した。LFM2.5は
    16層中6層だけがself_attn(残り10層はconv)、feed_forwardは16層全部にある
    (実際のnamed_modulesで確認済み: layers.N.self_attn.{q,k,v,out}_proj,
    layers.N.conv.{in,out}_proj, layers.N.feed_forward.{w1,w2,w3})。
    計算量の大半を占めるfeed_forward/conv層だけ量子化し、層数が少なく精度に
    効いていそうなself_attn層は全精度のまま残す(qconfig_specに量子化したい
    層の名前だけを明示的に列挙する形で、self_attnを除外する)。
    """
    if quantize and device != "cpu":
        raise ValueError("動的量子化はCPU向けなので device='cpu' と併用すること")

    model = load_model_from_hub(repo_id, device=device)
    if quantize:
        model.backbone = model.backbone.merge_and_unload()
        model = model.float()

        qconfig_spec = {
            name: torch.ao.quantization.default_dynamic_qconfig
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and "self_attn" not in name
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
