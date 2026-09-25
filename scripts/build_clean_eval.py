"""学習に一度も使っていないデータだけで評価セットを作り、bf16モデルで評価する。

学習には各データセットの train 分割だけを使っているので、次の未使用分割を評価に使う:
  - JCommonsenseQA: validation
  - chABSA: test (ミラー側の分割。同じ企業文書の別文が train にある可能性は残る)
  - JSNLI: dev.tsv
`train_val_split` による評価は、学習に渡した final_examples 自体から切り出しているため、
学習データと重なっていた(README参照)。こちらが本当のhold-out評価。

Colabでの例: python -m scripts.build_clean_eval build eval-bf16 push
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

from data.convert.schema import JevExample
from data.postprocess.shuffle_candidates import shuffle_all

REPO_ID = "fukayatti0/jev-japanese-judgment"
EVAL_PATH_IN_REPO = "eval/clean_eval.jsonl"


def build(n_per_task: int = 400, seed: int = 0, raw_dir: Path = Path("data/raw/jsnli")) -> list[JevExample]:
    from data.convert import chabsa, jcommonsenseqa, jsnli

    raw_dir.mkdir(parents=True, exist_ok=True)
    dev = raw_dir / "jsnli_1.1" / "dev.tsv"
    if not dev.exists():
        subprocess.run(
            ["curl", "-sL", "-o", str(raw_dir / "jsnli.zip"), "https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JSNLI/jsnli_1.1.zip"],
            check=True,
        )
        subprocess.run(["unzip", "-o", "-q", str(raw_dir / "jsnli.zip"), "-d", str(raw_dir)], check=True)

    pools = {
        "commonsense_qa": jcommonsenseqa.convert("validation"),
        "sentiment": chabsa.convert("test"),
        "nli": [ex.model_copy(update={"id": f"dev_{ex.id}"}) for ex in jsnli.convert(dev)],
    }
    rng = random.Random(seed)
    picked = []
    for task, pool in pools.items():
        chosen = rng.sample(pool, min(n_per_task, len(pool)))
        print(f"{task}: pool={len(pool)} picked={len(chosen)}")
        picked.extend(chosen)
    return shuffle_all(picked)


def save(examples: list[JevExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ex.model_dump_json() for ex in examples) + "\n", encoding="utf-8")


def load(path: Path) -> list[JevExample]:
    return [JevExample(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _metrics(model, examples: list[JevExample]) -> dict:
    from eval.calibration import brier_score, expected_calibration_error, negative_log_likelihood
    from eval.run_eval import collect_logits, evaluate

    acc = evaluate(model, examples)
    logits, labels = collect_logits(model, examples)
    return {
        **acc,
        "ece": expected_calibration_error(logits, labels),
        "brier": brier_score(logits, labels),
        "nll": negative_log_likelihood(logits, labels),
    }


def eval_bf16(examples: list[JevExample]) -> None:
    from data.postprocess.split import train_val_split
    from eval.run_eval import load_model_from_hub
    from scripts.push_to_hub import pull

    model = load_model_from_hub(REPO_ID, device="cuda").eval()
    print("CLEAN (未使用データ):", _metrics(model, examples))
    _, old_val = train_val_split(pull(repo_id=REPO_ID), val_size=500, seed=0)
    print("OLD (学習と重なる500件):", _metrics(model, old_val))


def push(path: Path) -> None:
    from huggingface_hub import HfApi

    from scripts.push_to_hub import _get_hf_token

    HfApi(token=_get_hf_token()).upload_file(
        path_or_fileobj=str(path), path_in_repo=EVAL_PATH_IN_REPO, repo_id=REPO_ID, repo_type="model"
    )
    print(f"pushed -> {EVAL_PATH_IN_REPO}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["build", "eval-bf16", "push"])
    ap.add_argument("--n-per-task", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("data/clean_eval.jsonl"))
    a = ap.parse_args()
    if "build" in a.stages:
        save(build(a.n_per_task), a.out)
    examples = load(a.out)
    print(f"clean eval set: {len(examples)} examples from {a.out}")
    if "eval-bf16" in a.stages:
        eval_bf16(examples)
    if "push" in a.stages:
        push(a.out)


if __name__ == "__main__":
    _main()
