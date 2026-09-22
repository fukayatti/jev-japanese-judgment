"""確率のキャリブレーション評価: ECE, Brier score, NLL, reliability diagram用の集計。"""

import torch
import torch.nn.functional as F


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    probs = torch.softmax(logits, dim=-1)
    confidences, preds = probs.max(dim=-1)
    accuracies = (preds == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.float().mean()) * torch.abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    one_hot = F.one_hot(labels, num_classes=probs.size(-1)).float()
    return float(((probs - one_hot) ** 2).sum(dim=-1).mean())


def negative_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, labels))


def reliability_diagram_data(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> list[dict]:
    probs = torch.softmax(logits, dim=-1)
    confidences, preds = probs.max(dim=-1)
    accuracies = (preds == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum())
        bins.append(
            {
                "bin_range": (float(lo), float(hi)),
                "count": count,
                "avg_confidence": float(confidences[in_bin].mean()) if count else None,
                "avg_accuracy": float(accuracies[in_bin].mean()) if count else None,
            }
        )
    return bins
