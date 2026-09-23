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

from data.convert.schema import JevExample
from eval.run_eval import load_model_from_hub, predict

DEFAULT_REPO_ID = "fukayatti0/jev-japanese-judgment"


def load(repo_id: str = DEFAULT_REPO_ID, device: str = "cuda"):
    return load_model_from_hub(repo_id, device=device)


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
    args = parser.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    model = load(args.repo_id, device=args.device)
    result = ask(model, args.question, candidates, context=args.context, device=args.device)

    print(f"question: {result['question']}")
    print(f"predicted: {result['predicted']}")
    for candidate, prob in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {candidate}: {prob:.4f}")


if __name__ == "__main__":
    _main()
