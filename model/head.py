"""Jevヘッド: LFM2.5-1.2B-JPのバックボーンに候補スコアリングヘッドを載せたモデル。

lm_head(自己回帰の生成ヘッド)は使わず、Lfm2Modelバックボーンの
最終トークンのhidden_stateを候補ごとにスコア化し、候補間でsoftmaxを取って
確率分布を直接出力する(非自己回帰)。

候補数Kは可変。各候補は "[context][SEP][question][SEP][candidate_i]" として
別々にエンコードし、(batch * K, seq_len) にまとめて1回のforwardで処理する。
"""

import torch
import torch.nn as nn
from transformers import AutoModel

BASE_MODEL_NAME = "LiquidAI/LFM2.5-1.2B-JP-202606"


class JevScoringHead(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        # pooled_hidden: (batch * K, hidden_size) -> (batch * K,)
        return self.mlp(pooled_hidden).squeeze(-1)


class JevModel(nn.Module):
    def __init__(self, base_model_name: str = BASE_MODEL_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        self.head = JevScoringHead(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids, attention_mask: (batch * K_max, seq_len) パディング済み。
        num_candidates: (batch,) 各サンプルの実際の候補数K_i (K_max以下)。
        戻り値: (batch, K_max) のマスク済みlogits(パディング分は-inf)。
        確率が欲しい場合は呼び出し側で softmax(dim=-1) すること
        (温度スケーリング等、softmax前のlogitsを直接使いたい場面があるため)。
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch*K_max, seq_len, hidden)

        # 各系列の最後の非パディングトークン位置を取得
        last_token_idx = attention_mask.sum(dim=1) - 1  # (batch*K_max,)
        pooled = last_hidden[torch.arange(last_hidden.size(0)), last_token_idx]  # (batch*K_max, hidden)

        scores = self.head(pooled)  # (batch*K_max,)

        batch_size = num_candidates.size(0)
        k_max = scores.size(0) // batch_size
        scores = scores.view(batch_size, k_max)

        candidate_mask = torch.arange(k_max, device=scores.device)[None, :] < num_candidates[:, None]
        scores = scores.masked_fill(~candidate_mask, float("-inf"))

        return scores
