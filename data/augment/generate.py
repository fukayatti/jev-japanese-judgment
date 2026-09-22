"""Colab上でvLLMを使い、拡張候補(言い換え/自然なリライト/ダミー選択肢)をバッチ生成するスクリプト。

ローカル環境では実行しない想定 (vllmはColab専用、requirements.txtではコメントアウト)。
実際のvLLM推論は data/augment/vllm_worker.py を subprocess として起動して行う
(ノートブックセル内で直接 vllm.LLM(...) を呼ぶとCUDA初期化済みプロセスからの
spawnでデッドロックすることがあるため)。
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from data.augment.prompts import DISTRACTOR_PROMPT, PARAPHRASE_PROMPT, REWRITE_NATURAL_PROMPT
from data.convert.schema import JevExample

# "Qwen/Qwen3-4B-Instruct" は存在しないリポジトリ名だった(404)。
# 正しいIDは "Qwen/Qwen3.5-4B" (Qwen3.5-4B-Baseのchat/instructチューン版、Apache 2.0)。
# image-text-to-text対応モデルだが、テキストのみのプロンプトでも問題なく使える。
AUGMENT_MODEL = "Qwen/Qwen3.5-4B"  # or Sarashina2.2, or LFM2.5-1.2B-JP


def build_prompts(examples: list[JevExample]) -> list[dict]:
    """各exampleに対して、言い換え/自然リライト/ダミー選択肢の3種のプロンプトを作る。"""
    jobs = []
    for ex in examples:
        jobs.append({"kind": "paraphrase", "example_id": ex.id, "prompt": PARAPHRASE_PROMPT.format(question=ex.question)})
        jobs.append({"kind": "rewrite_natural", "example_id": ex.id, "prompt": REWRITE_NATURAL_PROMPT.format(question=ex.question)})
        jobs.append(
            {
                "kind": "distractor",
                "example_id": ex.id,
                "prompt": DISTRACTOR_PROMPT.format(
                    context=ex.context,
                    question=ex.question,
                    correct_answer=ex.candidates[ex.label],
                    existing_candidates=", ".join(ex.candidates),
                ),
            }
        )
    return jobs


def run_batch_generation(jobs: list[dict], model_name: str = AUGMENT_MODEL) -> list[dict]:
    """vllm_worker.pyをsubprocessとして起動し、結果をJSON経由で受け取る。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "jobs.json"
        output_path = Path(tmpdir) / "results.json"

        with input_path.open("w", encoding="utf-8") as f:
            json.dump(jobs, f, ensure_ascii=False)

        proc = subprocess.run(
            [sys.executable, "-m", "data.augment.vllm_worker", str(input_path), str(output_path), model_name],
            capture_output=True,
            text=True,
        )
        # stdout/stderrはそのまま表示しつつ、失敗時は末尾をエラーメッセージにも含める
        # (Colabのセル出力だと何が原因か分かりにくいことがあるため)
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-30:])
            raise RuntimeError(f"vllm_worker failed (exit={proc.returncode}). stderr tail:\n{tail}")

        with output_path.open(encoding="utf-8") as f:
            return json.load(f)


if __name__ == "__main__":
    # Colab上で: examples を data/convert/*.py で作成した後にここに渡す
    raise SystemExit("Colabノートブックから examples を渡して呼び出すこと")
