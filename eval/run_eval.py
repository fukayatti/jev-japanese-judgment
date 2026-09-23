"""チェックポイントを読み込んでホールドアウトセットでaccuracy/calibrationを測る。

エポックの途中でも"latest"チェックポイントで精度を確認できるようにするための
軽量な評価ドライバ。
"""

from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.convert.schema import JevExample
from model.data_collator import JevDataCollator
from model.head import BASE_MODEL_NAME
from train import build_model, load_checkpoint


def load_model_from_checkpoint(checkpoint_dir, tag: str, device: str = "cuda"):
    model = build_model(use_gradient_checkpointing=False)
    load_checkpoint(model, checkpoint_dir, tag)
    return model.to(device)


@torch.no_grad()
def predict(model, examples: list[JevExample], device: str = "cuda", max_length: int = 256) -> list[dict]:
    """1件〜数件のexampleについて、候補ごとの確率を返す。手作業での動作確認用。"""
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    collator = JevDataCollator(tokenizer, max_length=max_length)
    batch = collator(examples)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    num_candidates = batch["num_candidates"].to(device)

    logits = model(input_ids, attention_mask, num_candidates)
    probs = torch.softmax(logits.float(), dim=-1).cpu()

    results = []
    for i, ex in enumerate(examples):
        k = len(ex.candidates)
        candidate_probs = {c: float(probs[i, j]) for j, c in enumerate(ex.candidates[:k])}
        predicted = ex.candidates[int(probs[i, :k].argmax())]
        results.append(
            {
                "context": ex.context,
                "question": ex.question,
                "predicted": predicted,
                "probabilities": candidate_probs,
            }
        )
    return results


@torch.no_grad()
def evaluate(model, examples, device: str = "cuda", batch_size: int = 16, max_length: int = 256) -> dict:
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    collator = JevDataCollator(tokenizer, max_length=max_length)
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=collator)

    correct = 0
    total = 0
    task_correct: dict[str, int] = defaultdict(int)
    task_total: dict[str, int] = defaultdict(int)

    idx = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        num_candidates = batch["num_candidates"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, num_candidates)
        preds = logits.argmax(dim=-1)

        batch_examples = examples[idx : idx + len(labels)]
        idx += len(labels)

        for pred, label, ex in zip(preds.tolist(), labels.tolist(), batch_examples):
            is_correct = int(pred == label)
            correct += is_correct
            total += 1
            task_correct[ex.task_type] += is_correct
            task_total[ex.task_type] += 1

    result = {"overall_accuracy": correct / total, "n": total}
    for task in task_total:
        result[f"{task}_accuracy"] = task_correct[task] / task_total[task]
        result[f"{task}_n"] = task_total[task]
    return result


# データセット全体での最大候補数(JCommonsenseQAが5択、chABSA/JSNLIは3値)。
@torch.no_grad()
def collect_logits(
    model, examples, device: str = "cuda", batch_size: int = 16, max_length: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """calibration指標を計算するため、全examplesのlogitsとlabelsを1つのテンソルに集約する。
    バッチごとにK_maxが異なりうる(候補数3のexampleだけのバッチ、5のexampleを含むバッチ、
    distractor拡張で6になったexample等)ので、examples全体の実際の最大候補数まで-infで
    パディングしてから結合する(-infはsoftmaxで確率0になるので、ECE/Brier/NLLの計算結果は
    変わらない)。固定値で決め打ちすると、拡張データで候補数が増えた場合に壊れる
    (実際にJCommonsenseQA+distractorで6択になるケースで形状エラーが発生した)。
    """
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    collator = JevDataCollator(tokenizer, max_length=max_length)
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=collator)

    global_max_k = max(len(ex.candidates) for ex in examples)

    all_logits = []
    all_labels = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        num_candidates = batch["num_candidates"].to(device)
        labels = batch["labels"]

        logits = model(input_ids, attention_mask, num_candidates).float().cpu()
        k = logits.size(1)
        if k < global_max_k:
            pad = torch.full((logits.size(0), global_max_k - k), float("-inf"))
            logits = torch.cat([logits, pad], dim=1)

        all_logits.append(logits)
        all_labels.append(labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)
