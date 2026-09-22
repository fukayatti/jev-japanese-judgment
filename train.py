"""JevModelの学習スクリプト。listwise cross-entropyでヘッドとLoRAアダプタを学習する。"""

from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
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


def build_model(use_gradient_checkpointing: bool = True) -> JevModel:
    model = JevModel(BASE_MODEL_NAME)
    model.backbone = get_peft_model(model.backbone, LORA_CONFIG)
    # LoRAだけ学習でも、勾配を通すには16層分のforward時の中間活性化を全部
    # 保持する必要があり、batch_size×候補数(最大5)を束ねると意外とVRAMを
    # 食う(T4でCUDA OOMを確認済み)。gradient checkpointingで再計算に倒して
    # メモリを節約できるが、再計算コストで学習が遅くなる。VRAMに余裕がある
    # GPU(L4/A100等)では不要なのでオフにできるようにしておく。
    # 有効にする場合、PEFTでフリーズしたバックボーンに対してenable_input_
    # require_grads()も必要(これが無いと勾配がLoRA層まで伝播しないことがある)。
    if use_gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()
    return model


def save_checkpoint(model: JevModel, checkpoint_dir: Path, tag: str) -> None:
    """学習対象(LoRAアダプタ+自作ヘッド)だけを保存する。凍結済みバックボーン本体は
    保存不要(容量の無駄、かつHubから再ダウンロードできる)。
    optimizerの状態は保存しない(再開時はoptimizerの運動量情報がリセットされる、
    という制約はあるが、セッション切断で全部消えるよりはずっとまし)。
    """
    path = checkpoint_dir / tag
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path / "lora")
    torch.save(model.head.state_dict(), path / "head.pt")


def load_checkpoint(model: JevModel, checkpoint_dir: Path, tag: str) -> None:
    from peft import PeftModel

    path = checkpoint_dir / tag
    model.backbone = PeftModel.from_pretrained(model.backbone.get_base_model(), path / "lora", is_trainable=True)
    model.head.load_state_dict(torch.load(path / "head.pt", map_location="cpu"))


def train_one_epoch(
    model: JevModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    checkpoint_dir: Path | None = None,
    checkpoint_every_steps: int = 200,
) -> float:
    model.train()
    total_loss = 0.0
    # epoch単位でしか出力がないと、1エポックがGPUでも数分〜数十分かかる規模の
    # データ(数万件)では「本当にハングしているのか、ただ時間がかかっているだけか」
    # 区別がつかない。バッチ単位の進捗バーで可視化する。
    progress = tqdm(dataloader, desc="train", unit="batch")
    for step, batch in enumerate(progress, start=1):
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
        progress.set_postfix(loss=f"{total_loss / step:.4f}")

        # 1エポックが数時間かかりうる規模なので、エポック境界だけの保存だと
        # Colabのセッション切断で数時間分の進捗が丸ごと消えかねない。
        # "latest"を定期的に上書き保存し、ディスク使用量も一定に保つ。
        if checkpoint_dir is not None and step % checkpoint_every_steps == 0:
            save_checkpoint(model, checkpoint_dir, "latest")

    return total_loss / len(dataloader)


def main(
    train_examples,
    val_examples=None,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-4,
    checkpoint_dir: str | None = None,
    checkpoint_every_steps: int = 200,
    use_gradient_checkpointing: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    # 実際に流れるのは batch_size × 候補数(最大5) 系列なので、512は必要以上に
    # 大きい(実データの質問文・chABSAの文はほとんどこれよりずっと短い)。
    collator = JevDataCollator(tokenizer, max_length=256)

    model = build_model(use_gradient_checkpointing=use_gradient_checkpointing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(train_examples, batch_size=batch_size, shuffle=True, collate_fn=collator)
    print(f"training on {len(train_examples)} examples, {len(train_loader)} batches/epoch, device={device}")

    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt_path:
        print(f"checkpointing to {ckpt_path} every {checkpoint_every_steps} steps and at each epoch end")

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, ckpt_path, checkpoint_every_steps)
        print(f"epoch {epoch + 1}/{epochs} loss={avg_loss:.4f}")
        if ckpt_path:
            save_checkpoint(model, ckpt_path, f"epoch{epoch + 1}")

    return model


if __name__ == "__main__":
    raise SystemExit("train_examples/val_examples を用意してから main() を呼び出すこと")
