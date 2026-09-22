"""温度スケーリングによる事後キャリブレーション。

学習済みモデルのスコア(softmax前のlogit)に対して、検証データ上で
NLLを最小化する温度パラメータTを1つだけ学習する。
"""

import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, steps: int = 200) -> float:
    """logits: (N, K) マスク済み(パディング分は-inf)。labels: (N,)。"""
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=steps)

    def closure():
        optimizer.zero_grad()
        scaled = logits / temperature
        loss = F.cross_entropy(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)
