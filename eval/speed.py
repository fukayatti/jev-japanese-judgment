"""推論レイテンシ計測: Jevヘッド方式 vs 自己回帰JSON生成方式。"""

import time

import torch


def measure_head_latency(model, collator, examples, device: str, n_repeats: int = 3) -> float:
    batch = collator(examples)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    num_candidates = batch["num_candidates"].to(device)

    model.eval()
    with torch.no_grad():
        # ウォームアップ
        model(input_ids, attention_mask, num_candidates)
        torch.cuda.synchronize() if device == "cuda" else None

        start = time.perf_counter()
        for _ in range(n_repeats):
            model(input_ids, attention_mask, num_candidates)
        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - start

    return elapsed / n_repeats / len(examples)


def measure_autoregressive_latency(pipeline, prompts: list[str], n_repeats: int = 3) -> float:
    """pipeline: transformers.pipeline("text-generation", ...) のようなchat推論呼び出し。"""
    # ウォームアップ
    pipeline(prompts[0])

    start = time.perf_counter()
    for _ in range(n_repeats):
        for prompt in prompts:
            pipeline(prompt)
    elapsed = time.perf_counter() - start

    return elapsed / n_repeats / len(prompts)
