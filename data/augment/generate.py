"""Colab上でvLLMを使い、拡張候補(言い換え/自然なリライト/ダミー選択肢)をバッチ生成するスクリプト。

ローカル環境では実行しない想定 (vllmはColab専用、requirements.txtではコメントアウト)。
guided decodingのAPI名はvLLMのバージョンで変わることがあるので、
実行時にインストール済みバージョンのドキュメントで要確認。
"""

from pydantic import BaseModel

from data.augment.prompts import DISTRACTOR_PROMPT, PARAPHRASE_PROMPT, REWRITE_NATURAL_PROMPT
from data.convert.schema import JevExample

# "Qwen/Qwen3-4B-Instruct" は存在しないリポジトリ名だった(404)。
# 正しいIDは "Qwen/Qwen3.5-4B" (Qwen3.5-4B-Baseのchat/instructチューン版、Apache 2.0)。
# image-text-to-text対応モデルだが、テキストのみのプロンプトでも問題なく使える。
AUGMENT_MODEL = "Qwen/Qwen3.5-4B"  # or Sarashina2.2, or LFM2.5-1.2B-JP


class _TextOutput(BaseModel):
    text: str


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
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, gpu_memory_utilization=0.9, max_model_len=2048)
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=256,
        guided_decoding=_TextOutput.model_json_schema(),
    )
    outputs = llm.generate([job["prompt"] for job in jobs], sampling_params)
    results = []
    for job, output in zip(jobs, outputs):
        results.append({**job, "output": output.outputs[0].text})
    return results


if __name__ == "__main__":
    # Colab上で: examples を data/convert/*.py で作成した後にここに渡す
    raise SystemExit("Colabノートブックから examples を渡して呼び出すこと")
