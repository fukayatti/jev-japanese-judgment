"""GGUF版のJevモデルに質問を投げる(torch/transformers/peft不要、CPUのみ)。

必要なもの: numpy, huggingface_hub, ビルド済みの llama.cpp の `llama-embedding`
(PATH上に置くか --llama-embedding で指定)。

使い方:
  python -m scripts.ask_gguf --question "日本の首都はどこ？" --candidates "大阪,東京,京都,名古屋"
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from scripts.gguf_pipeline import REPO_ID, GGUFScorer


def load(quant: str = "Q4_K_M", repo_id: str = REPO_ID, llama_embedding: str | None = None, threads: int = 4) -> GGUFScorer:
    binary = llama_embedding or shutil.which("llama-embedding")
    if binary is None:
        raise FileNotFoundError("llama-embedding が見つからない。llama.cppをビルドしてPATHに通すか --llama-embedding で指定すること")
    fetch = lambda name: Path(hf_hub_download(repo_id, f"gguf/{name}"))
    return GGUFScorer(fetch(f"jev-{quant}.gguf"), None, Path(binary), fetch("head.npz"), threads)


def ask(scorer: GGUFScorer, question: str, candidates: list[str], context: str = "") -> dict:
    texts = [f"{context}\n{question}\n{c}" for c in candidates]
    scores = scorer.head(scorer.embed(texts))
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    return {
        "question": question,
        "predicted": candidates[int(scores.argmax())],
        "probabilities": dict(zip(candidates, probs.tolist())),
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description="GGUF版Jevモデルに質問を投げる")
    ap.add_argument("--question", required=True)
    ap.add_argument("--candidates", required=True, help="カンマ区切りの候補一覧")
    ap.add_argument("--context", default="")
    ap.add_argument("--quant", default="Q4_K_M", choices=["Q4_0", "Q4_K_M", "Q8_0"])
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--llama-embedding", default=None, help="llama-embeddingバイナリのパス")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    candidates = [c.strip() for c in a.candidates.split(",") if c.strip()]
    result = ask(load(a.quant, a.repo_id, a.llama_embedding, a.threads), a.question, candidates, a.context)

    print(f"question: {result['question']}")
    print(f"predicted: {result['predicted']}")
    for c, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {c}: {p:.4f}")


if __name__ == "__main__":
    _main()
