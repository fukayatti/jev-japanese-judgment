"""チェックポイントを読み込んでホールドアウトセットでaccuracyを測る。

エポックの途中でも"latest"チェックポイントで精度を確認できるようにするための
軽量な評価ドライバ。calibration(ECE等)まではやらず、まずは正解率の確認用。
"""

from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.data_collator import JevDataCollator
from model.head import BASE_MODEL_NAME
from train import build_model, load_checkpoint


def load_model_from_checkpoint(checkpoint_dir, tag: str, device: str = "cuda"):
    model = build_model(use_gradient_checkpointing=False)
    load_checkpoint(model, checkpoint_dir, tag)
    return model.to(device)


@torch.no_grad()
def evaluate(model, examples, device: str = "cuda", batch_size: int = 16, max_length: int = 256) -> dict:
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    collator = JevDataCollator(tokenizer, max_length=max_length)
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=collator)

    correct = 0
    total = 0
    task_correct: dict[str, int] = defaultdict(int)
    task_total: dict[str, int] = defaultdict(int)

    idx = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        num_candidates = batch["num_candidates"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, num_candidates)
        preds = logits.argmax(dim=-1)

        batch_examples = examples[idx : idx + len(labels)]
        idx += len(labels)

        for pred, label, ex in zip(preds.tolist(), labels.tolist(), batch_examples):
            is_correct = int(pred == label)
            correct += is_correct
            total += 1
            task_correct[ex.task_type] += is_correct
            task_total[ex.task_type] += 1

    result = {"overall_accuracy": correct / total, "n": total}
    for task in task_total:
        result[f"{task}_accuracy"] = task_correct[task] / task_total[task]
        result[f"{task}_n"] = task_total[task]
    return result
