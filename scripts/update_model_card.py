"""HFモデルカード(README.md)全体を、実測値から再生成して上書きする。

チェックポイントの再アップロードは不要(README.mdだけ更新する)。数値の出所:
  - CLEAN_BF16: scripts/build_clean_eval.py eval-bf16 (未使用データ1200件、bf16)
  - QUANTIZED_INT8: PyTorch動的int8(per-channel)。学習データと重なる200件での測定
  - GGUF_MERGED: scripts/gguf_pipeline.py eval --merged --eval-file (未使用データ1200件)

使い方:
  python -m scripts.update_model_card --dry-run   # ローカルでREADMEを出力して確認
  python -m scripts.update_model_card             # HFへ反映(HF_TOKENが必要)
"""

import argparse
from pathlib import Path

from scripts.gguf_pipeline import render_gguf_section
from scripts.push_model_to_hub import render_model_card, render_quantized_section

REPO_ID = "fukayatti0/jev-japanese-judgment"

CLEAN_BF16 = {
    "n": 1200,
    "overall_accuracy": 0.9216666666666666,
    "commonsense_qa_accuracy": 0.9175,
    "commonsense_qa_n": 400,
    "sentiment_accuracy": 0.925,
    "sentiment_n": 400,
    "nli_accuracy": 0.9225,
    "nli_n": 400,
    "ece": 0.014452877454459667,
    "brier": 0.1179094985127449,
    "nll": 0.22250786423683167,
}

QUANTIZED_INT8 = {
    "n": 200,
    "overall_accuracy": 0.895,
    "commonsense_qa_accuracy": 0.8888888888888888,
    "commonsense_qa_n": 81,
    "nli_accuracy": 0.8734177215189873,
    "nli_n": 79,
    "sentiment_accuracy": 0.95,
    "sentiment_n": 40,
}

GGUF_MERGED = {
    "Q4_0": {"accuracy": 0.918, "ece": 0.021, "brier": 0.1216, "nll": 0.2287},
    "Q4_K_M": {"accuracy": 0.915, "ece": 0.0183, "brier": 0.1239, "nll": 0.2345},
    "Q8_0": {"accuracy": 0.921, "ece": 0.0154, "brier": 0.1176, "nll": 0.2215},
}


def build_card() -> str:
    marker_q = "\n## 量子化版(ローカルPC/CPU向け)\n"
    card = render_model_card(REPO_ID, CLEAN_BF16)
    card += marker_q + render_quantized_section(REPO_ID, QUANTIZED_INT8)
    bf16 = {"accuracy": CLEAN_BF16["overall_accuracy"], "ece": CLEAN_BF16["ece"]}
    card += render_gguf_section(GGUF_MERGED, bf16, CLEAN_BF16["n"])
    return card


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="HFへ送らず、README.mdをローカルに書き出すだけ")
    ap.add_argument("--out", type=Path, default=Path("README_model_card.md"))
    a = ap.parse_args()

    card = build_card()
    if a.dry_run:
        a.out.write_text(card, encoding="utf-8")
        print(f"wrote {a.out} ({len(card)} chars)")
        return

    from huggingface_hub import HfApi

    from scripts.push_to_hub import _get_hf_token

    HfApi(token=_get_hf_token()).upload_file(
        path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=REPO_ID, repo_type="model"
    )
    print("updated README.md on", REPO_ID)


if __name__ == "__main__":
    main()
