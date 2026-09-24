"""feed_forward/conv層をint8量子化に強くするためのQAT(Quantization-Aware Training)。

背景: PTQ(学習後量子化)を試した結果、held-out 200件でoverall accuracyが
97.0%(bf16) -> 89.0%(全層int8) / 86.5%(self_attn除く選択的int8)と、
どちらも大きく劣化することを実測で確認した。選択的量子化はself_attnを
守ったにも関わらず全体ではむしろ悪化しており、単純なPTQでは頭打りと判断。

方針: epoch1チェックポイント(LoRA+ヘッド)を読み込み、LoRAをバックボーンへ
マージした上で、量子化対象の層(feed_forward.w1/w2/w3, conv.in_proj/
out_proj)だけ凍結解除してfake quantizationを仕込みながら再学習する。
self_attn層とヘッドは凍結したまま(推論時のself_attnは全精度のまま使う
設計のため)。

注意: feed_forward層だけでモデル全体の大半のパラメータ(数億〜10億規模)を
占めるため、LoRAだった元の学習(324万パラメータ)とは規模が違う。小さい
学習率・少ないステップ数に留め、gradient checkpointingも併用すること。
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from eval.run_eval import load_model_from_checkpoint
from model.data_collator import JevDataCollator
from model.head import BASE_MODEL_NAME, JevModel

QAT_BACKEND = "qnnpack"  # ARM系。x86 CPUなら'x86'に変えること


def prepare_qat_model(model: JevModel, backend: str = QAT_BACKEND) -> JevModel:
    """LoRAをマージし、量子化対象層だけ凍結解除+fake quant準備する。それ以外は全部凍結。"""
    model.backbone = model.backbone.merge_and_unload()
    model = model.float()

    for p in model.parameters():
        p.requires_grad = False

    qat_qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    target_count = 0
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Linear) and "self_attn" not in name:
            module.qconfig = qat_qconfig
            target_count += 1
            for p in module.parameters():
                p.requires_grad = True

    print(f"QAT target linear layers: {target_count}")
    model.backbone.train()  # prepare_qatはtrainingモード必須
    model.backbone = torch.ao.quantization.prepare_qat(model.backbone, inplace=False)
    model.backbone.gradient_checkpointing_enable()
    model.backbone.enable_input_require_grads()
    return model


def train_qat(
    model: JevModel,
    train_examples,
    steps: int = 500,
    batch_size: int = 8,
    lr: float = 1e-5,
    device: str = "cuda",
    checkpoint_dir: str | None = None,
) -> JevModel:
    """フルepochではなく、ステップ数で区切る(QATは量子化ノイズへの適応が目的で、
    タスクをゼロから学び直すわけではないため、原則として元の学習より遥かに
    少ないステップ数で十分なはず)。
    """
    model.to(device)
    model.train()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    collator = JevDataCollator(tokenizer, max_length=256)
    loader = DataLoader(train_examples, batch_size=batch_size, shuffle=True, collate_fn=collator)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"trainable params: {n_trainable:,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    step = 0
    total_loss = 0.0
    progress = tqdm(total=steps, desc="qat")
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            num_candidates = batch["num_candidates"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, num_candidates)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            total_loss += loss.item()
            progress.update(1)
            progress.set_postfix(loss=f"{total_loss / step:.4f}")

            if checkpoint_dir is not None and step % 100 == 0:
                _save_qat_checkpoint(model, checkpoint_dir, "qat_latest", step)
    progress.close()
    return model


def _save_qat_checkpoint(model: JevModel, checkpoint_dir: str, tag: str, step: int) -> None:
    path = Path(checkpoint_dir) / tag
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "meta.json").write_text(json.dumps({"step": step}))


def convert_qat_model(model: JevModel) -> JevModel:
    """fake quantizeで学習した重みを、実際のint8量子化モデルに変換する。"""
    model.eval()
    model.backbone = torch.ao.quantization.convert(model.backbone.eval(), inplace=False)
    return model


if __name__ == "__main__":
    raise SystemExit("train_examples を用意してから prepare_qat_model/train_qat/convert_qat_model を呼び出すこと")
