"""学習に一度も使っていないデータだけで評価セットを作り、bf16モデルで評価する。

学習には各データセットの train 分割だけを使っているので、次の未使用分割を評価に使う:
  - JCommonsenseQA: validation
  - chABSA: test (ミラー側の分割。同じ企業文書の別文が train にある可能性は残る)
  - JSNLI: dev.tsv
`train_val_split` による評価は、学習に渡した final_examples 自体から切り出しているため、
学習データと重なっていた(README参照)。こちらが本当のhold-out評価。

Colabでの例: python -m scripts.build_clean_eval build eval-bf16 push
追加タスク(noul/score)用: python -m scripts.build_clean_eval build-extras eval-ckpt --out data/clean_eval_extras.jsonl \\
    --checkpoint-dir <dir> --tag epoch1
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


def build_extras(seed: int = 0) -> list[JevExample]:
    """noul/score用の未使用データ評価セット。noulはyes/noを半々(200+200)にして、多数派に寄せるだけで
    高得点になる(JNLIは「いいえ」86%)のを防ぐ。task_typeはデータセットごとに変えて、別々に集計できるようにする。"""
    from data.convert import jcola_noul, jnli_noul, jsick_score, jsts_score
    from scripts.build_judgment_mix import balance_binary

    def take_balanced(examples, per_class):
        balanced = balance_binary(examples, seed)
        rng = random.Random(seed)
        by_answer: dict[str, list] = {}
        for ex in balanced:
            by_answer.setdefault(ex.candidates[ex.label], []).append(ex)
        return [ex for v in by_answer.values() for ex in rng.sample(v, min(per_class, len(v)))]

    rng = random.Random(seed)
    jcola_valid = jcola_noul.convert("in_domain_valid") + jcola_noul.convert("out_of_domain_valid")
    parts = {
        "jnli_noul": take_balanced(jnli_noul.convert("test"), 200),
        "jcola_noul": take_balanced(jcola_valid, 200),
        "jsts_score": rng.sample(jsts_score.convert("validation"), 400),
        "jsick_score": rng.sample(jsick_score.convert("test"), 400),
    }
    picked = []
    for name, exs in parts.items():
        print(f"{name}: {len(exs)}")
        picked.extend(ex.model_copy(update={"task_type": name}) for ex in exs)
    return shuffle_all(picked)


def eval_checkpoint(examples: list[JevExample], checkpoint_dir: str, tag: str) -> None:
    from eval.run_eval import load_model_from_checkpoint

    model = load_model_from_checkpoint(checkpoint_dir, tag).eval()
    print(f"{checkpoint_dir}/{tag}:", _metrics(model, examples))


def build_calibration(eval_paths: list[Path], seed: int = 1, n_per_task: int = 300, raw_dir: Path = Path("data/raw/jsnli")) -> list[JevExample]:
    """温度スケーリングで温度を決めるための、評価セットとは重ならない未使用データ。
    評価セット(eval_paths)に入っているidを除いた残りから取る。学習に使っていない分割だけを使うので、
    Tを決めるのに使っても評価セットの数字は汚れない。"""
    from data.convert import chabsa, jcola_noul, jcommonsenseqa, jnli_noul, jsick_score, jsnli, jsts_score
    from scripts.build_judgment_mix import balance_binary

    exclude = {ex.id for p in eval_paths for ex in load(p)}
    rng = random.Random(seed)
    keep = lambda exs: [e for e in exs if e.id not in exclude]

    def take_balanced(exs, per_class):
        by_answer: dict[str, list] = {}
        for ex in balance_binary(keep(exs), seed):
            by_answer.setdefault(ex.candidates[ex.label], []).append(ex)
        return [ex for v in by_answer.values() for ex in rng.sample(v, min(per_class, len(v)))]

    dev = raw_dir / "jsnli_1.1" / "dev.tsv"
    if not dev.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "-o", str(raw_dir / "jsnli.zip"), "https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JSNLI/jsnli_1.1.zip"], check=True)
        subprocess.run(["unzip", "-o", "-q", str(raw_dir / "jsnli.zip"), "-d", str(raw_dir)], check=True)
    jcola_valid = jcola_noul.convert("in_domain_valid") + jcola_noul.convert("out_of_domain_valid")
    parts = {
        "commonsense_qa": keep(jcommonsenseqa.convert("validation")),
        "sentiment": keep(chabsa.convert("test")),
        "nli": keep([ex.model_copy(update={"id": f"dev_{ex.id}"}) for ex in jsnli.convert(dev)]),
        "jsts_score": keep(jsts_score.convert("validation")),
        "jsick_score": keep(jsick_score.convert("test")),
    }
    picked = []
    for name, pool in parts.items():
        chosen = rng.sample(pool, min(n_per_task, len(pool)))
        picked.extend(ex.model_copy(update={"task_type": name}) for ex in chosen)
    for name, exs in (("jnli_noul", jnli_noul.convert("test")), ("jcola_noul", jcola_valid)):
        picked.extend(ex.model_copy(update={"task_type": name}) for ex in take_balanced(exs, n_per_task // 2))
    for name in {e.task_type for e in picked}:
        print(f"calibration {name}: {sum(e.task_type == name for e in picked)}")
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

    in_repo = f"eval/{path.name}"
    HfApi(token=_get_hf_token()).upload_file(
        path_or_fileobj=str(path), path_in_repo=in_repo, repo_id=REPO_ID, repo_type="model"
    )
    print(f"pushed -> {in_repo}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["build", "build-extras", "build-calib", "eval-bf16", "eval-ckpt", "push"])
    ap.add_argument("--n-per-task", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("data/clean_eval.jsonl"))
    ap.add_argument("--checkpoint-dir", default=None, help="eval-ckptで評価するチェックポイントの置き場所")
    ap.add_argument("--tag", default="epoch1", help="eval-ckptで評価するチェックポイントのtag")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[Path("data/clean_eval.jsonl"), Path("data/clean_eval_extras.jsonl")],
                    help="build-calibで除外する評価セット(idが重ならないようにする)")
    a = ap.parse_args()
    if "build" in a.stages:
        save(build(a.n_per_task), a.out)
    if "build-extras" in a.stages:
        save(build_extras(), a.out)
    if "build-calib" in a.stages:
        save(build_calibration(a.exclude), a.out)
    examples = load(a.out)
    print(f"clean eval set: {len(examples)} examples from {a.out}")
    if "eval-bf16" in a.stages:
        eval_bf16(examples)
    if "eval-ckpt" in a.stages:
        eval_checkpoint(examples, a.checkpoint_dir, a.tag)
    if "push" in a.stages:
        push(a.out)


if __name__ == "__main__":
    _main()
