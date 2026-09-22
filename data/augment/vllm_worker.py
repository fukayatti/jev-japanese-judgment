"""vLLMバッチ生成の実処理。単体の.pyスクリプトとして別プロセスで実行する。

Jupyter/Colabのノートブックセル内で直接 vllm.LLM(...) を呼ぶと、
ノートブックのカーネルプロセスが既にCUDAを初期化済みの状態でvLLMが
ワーカーをspawnしようとしてデッドロックすることがある
(vLLM公式のtroubleshootingにも記載されているノートブック特有の問題)。
それを避けるため、生成処理は必ずこのスクリプトを subprocess で
起動する形にする (呼び出し元は data/augment/generate.py)。

使い方: python -m data.augment.vllm_worker <input_jobs.json> <output_results.json>
"""

import json
import sys

from pydantic import BaseModel


class _TextOutput(BaseModel):
    text: str


def main(input_path: str, output_path: str, model_name: str) -> None:
    from vllm import LLM, SamplingParams

    with open(input_path, encoding="utf-8") as f:
        jobs = json.load(f)

    llm = LLM(model=model_name, gpu_memory_utilization=0.9, max_model_len=2048)
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=256,
        guided_decoding=_TextOutput.model_json_schema(),
    )
    outputs = llm.generate([job["prompt"] for job in jobs], sampling_params)

    results = [{**job, "output": output.outputs[0].text} for job, output in zip(jobs, outputs)]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    input_path, output_path, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
    main(input_path, output_path, model_name)
