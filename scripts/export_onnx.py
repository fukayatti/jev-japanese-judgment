"""学習済みJevモデルをONNX形式にエクスポートする(ローカルPCでの軽量実行・量子化用)。

設計方針: 「候補ごとのスコアを出す」重い計算(バックボーン+ヘッド)だけを
ONNXにエクスポートし、「候補間のsoftmax」は呼び出し側(ONNX Runtime利用側)の
軽い後処理として扱う。候補数が可変な部分のロジックをグラフに含めずに済み、
エクスポートの成功率が上がる(cross-encoder型のリランカーでよく使われる構成)。

LoRAはエクスポート前にバックボーンへマージする(PEFTラッパーのままだと
トレースが複雑になるため)。
"""

import torch
import torch.nn as nn

from eval.run_eval import load_model_from_hub
from model.head import JevScoringHead


class ExportableJevModel(nn.Module):
    """torch.onnx.exportでトレースしやすいよう、pure tensor opsだけのforwardに整理したラッパー。
    入力: [context][question][candidate_i] をエンコードした1系列
    出力: その候補のスコア(スカラー、バッチ分)
    """

    def __init__(self, backbone, head: JevScoringHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_token_idx = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[torch.arange(last_hidden.size(0)), last_token_idx]
        return self.head(pooled)


def export(
    repo_id: str = "fukayatti0/jev-japanese-judgment",
    output_path: str = "jev_model.onnx",
    opset: int = 17,
) -> None:
    model = load_model_from_hub(repo_id, device="cpu")
    merged_backbone = model.backbone.merge_and_unload()

    export_model = ExportableJevModel(merged_backbone, model.head)
    export_model.eval()

    dummy_input_ids = torch.randint(0, 1000, (4, 16), dtype=torch.long)
    dummy_attention_mask = torch.ones(4, 16, dtype=torch.long)

    # レガシーのTorchScriptベースのtracerは、LFM2の一部カスタムカーネル
    # (融合RMSNorm/conv層のautograd関数)でRuntimeError: unordered_map::at と
    # いうC++レベルのエラーを起こして失敗した。dynamo(FXグラフキャプチャ)
    # ベースのエクスポーターの方が、こうしたカスタムopとの相性が良いことがある。
    torch.onnx.export(
        export_model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["scores"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "scores": {0: "batch"},
        },
        opset_version=opset,
        dynamo=True,
    )
    print(f"exported to {output_path}")


if __name__ == "__main__":
    export()
