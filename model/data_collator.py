"""JevExampleのバッチを、候補ごとに展開してトークナイズするcollator。"""

import torch
from transformers import PreTrainedTokenizerBase

from data.convert.schema import JevExample


class JevDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.tokenizer = tokenizer
        # model/head.py の最終トークン抽出 (attention_mask.sum(dim=1) - 1) は
        # 右パディング前提。デコーダ系トークナイザーは生成用に左パディングが
        # デフォルトのことがあるため、ここで明示的に右パディングを強制する。
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            # デコーダ系トークナイザーはpad_token未設定のことが多い
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __call__(self, examples: list[JevExample]) -> dict:
        num_candidates = torch.tensor([len(ex.candidates) for ex in examples])
        k_max = int(num_candidates.max())

        flat_texts = []
        for ex in examples:
            for i in range(k_max):
                if i < len(ex.candidates):
                    text = f"{ex.context}\n{ex.question}\n{ex.candidates[i]}"
                else:
                    text = ""  # パディング用ダミー候補、attention_maskで無視される
                flat_texts.append(text)

        encoded = self.tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = torch.tensor([ex.label for ex in examples])

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "num_candidates": num_candidates,
            "labels": labels,
        }
