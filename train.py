"""JevModelの学習スクリプト。listwise cross-entropyでヘッドとLoRAアダプタを学習する。"""

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.data_collator import JevDataCollator
from model.head import BASE_MODEL_NAME, JevModel

# transformers.models.lfm2.modeling_lfm2 のソースで確認済みの実モジュール名。
# Lfm2Attention: q_proj, k_proj, v_proj, out_proj (Llama系の"o_proj"ではない)
# Lfm2ShortConv (conv層): in_proj, out_proj
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj"],
    bias="none",
)


def build_model() -> JevModel:
    model = JevModel(BASE_MODEL_NAME)
    model.backbone = get_peft_model(model.backbone, LORA_CONFIG)
    # LoRAだけ学習でも、勾配を通すには16層分のforward時の中間活性化を全部
    # 保持する必要があり、batch_size×候補数(最大5)を束ねると意外とVRAMを
    # 食う(T4でCUDA OOMを確認済み)。gradient checkpointingで再計算に倒して
    # メモリを節約する。PEFTでフリーズしたバックボーンに対して有効にするには
    # enable_input_require_grads()も必要(これが無いと勾配がLoRA層まで
    # 伝播しないことがある)。
    model.backbone.gradient_checkpointing_enable()
    model.backbone.enable_input_require_grads()
    return model


def train_one_epoch(model: JevModel, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        num_candidates = batch["num_candidates"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, num_candidates)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def main(train_examples, val_examples=None, epochs: int = 3, batch_size: int = 4, lr: float = 2e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    # 実際に流れるのは batch_size × 候補数(最大5) 系列なので、512は必要以上に
    # 大きい(実データの質問文・chABSAの文はほとんどこれよりずっと短い)。
    # OOM対策としてgradient checkpointingに加えてここも絞っておく。
    collator = JevDataCollator(tokenizer, max_length=256)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(train_examples, batch_size=batch_size, shuffle=True, collate_fn=collator)

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"epoch {epoch + 1}/{epochs} loss={avg_loss:.4f}")

    return model


if __name__ == "__main__":
    raise SystemExit("train_examples/val_examples を用意してから main() を呼び出すこと")
