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


def prepare_qat_model(model: JevModel) -> JevModel:
    """LoRAをマージし、量子化対象層だけ凍結解除+fake quant準備する。それ以外は全部凍結。

    get_default_qat_qconfig()(静的QAT)を使うと、convert後に quantized::linear
    (活性化も事前に量子化されている前提の静的演算)になり、CPU上でも
    "NotImplementedError: Could not run 'quantized::linear' with arguments
    from the 'CPU' backend" になった(QuantStub/DeQuantStubを挟んでいないため)。
    PTQで実績のある quantized::linear_dynamic に変換されるよう、
    default_dynamic_qat_qconfig(学習時は重みだけfake quant、活性化は動的)を使う。
    """
    model.backbone = model.backbone.merge_and_unload()
    model = model.float()

    for p in model.parameters():
        p.requires_grad = False

    qat_qconfig = torch.ao.quantization.default_dynamic_qat_qconfig
    target_count = 0
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Linear) and "self_attn" not in name:
            module.qconfig = qat_qconfig
            target_count += 1
            for p in module.parameters():
                p.requires_grad = True

    print(f"QAT target linear layers: {target_count}")
    model.backbone.train()  # prepare_qatはtrainingモード必須
    # prepare_qatのデフォルトmapping(DEFAULT_QAT_MODULE_MAPPINGS)は
    # nn.Linear -> nn.qat.Linear(静的QAT用)にしてしまう。動的QATにしたいので
    # nn.qat.dynamic.Linearを明示的に指定する。
    import torch.ao.nn.qat.dynamic as nnqatd

    qat_mapping = {torch.nn.Linear: nnqatd.Linear}
    model.backbone = torch.ao.quantization.prepare_qat(model.backbone, mapping=qat_mapping, inplace=False)
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
    """fake quantizeで学習した重みを、実際のint8量子化モデルに変換する。
    量子化バックエンド(CPU向け)は他のGPU上のテンソルと混在できないため、
    変換前にモデル全体をCPUへ移しておく(self_attn等の非量子化部分がCUDAに
    残ったままだと、evaluate時に"Expected all tensors to be on the same
    device"で落ちる)。

    convert()のデフォルトmappingは静的量子化用で、そのままだと
    quantized::linear(活性化側もQuantStub/DeQuantStubで事前量子化されている
    前提の演算)になり、"NotImplementedError: Could not run 'quantized::linear'
    ... from the 'CPU' backend"になる。PTQで実績のあるquantized::linear_dynamic
    に変換されるよう、動的量子化用mappingを明示的に指定する。
    """
    from torch.ao.quantization.quantization_mappings import get_default_dynamic_quant_module_mappings

    model = model.to("cpu")
    model.eval()
    dynamic_mapping = get_default_dynamic_quant_module_mappings()
    model.backbone = torch.ao.quantization.convert(model.backbone.eval(), mapping=dynamic_mapping, inplace=False)
    return model


if __name__ == "__main__":
    raise SystemExit("train_examples を用意してから prepare_qat_model/train_qat/convert_qat_model を呼び出すこと")
