"""タスク別accuracyと混同行列。"""

from collections import defaultdict

import torch
from sklearn.metrics import confusion_matrix


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean())


def compute_accuracy_by_task(
    logits: torch.Tensor, labels: torch.Tensor, task_types: list[str]
) -> dict[str, float]:
    grouped = defaultdict(list)
    preds = logits.argmax(dim=-1)
    for pred, label, task in zip(preds.tolist(), labels.tolist(), task_types):
        grouped[task].append(pred == label)

    result = {task: sum(correct) / len(correct) for task, correct in grouped.items()}
    result["macro_avg"] = sum(result.values()) / len(result)
    return result


def compute_confusion_matrix(logits: torch.Tensor, labels: torch.Tensor):
    preds = logits.argmax(dim=-1)
    return confusion_matrix(labels.tolist(), preds.tolist())
