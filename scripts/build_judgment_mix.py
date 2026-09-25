"""noul(yes/no)とscore(段階評価)のデータを、既存の3タスクと混ぜた追加学習用データセットを作る。

構成(既定値):
  - noul: JNLI + JCoLA。yes/noをソースごとに半々にそろえる(元は JNLI が「はい」14%、JCoLAが83%と逆向きに偏る)
  - score: JSTS(段階ごとに最大1500件)+ JSICK(段階ごとに最大800件)。多数派の段階だけ間引く
  - replay: 既存データ(JCommonsenseQA / chABSA / JSNLI + LLM拡張)から各ソース最大3400件。元の3タスクを忘れないため
評価用の分割(JNLI test, JCoLA valid, JSTS validation, JSICK test)は絶対に混ぜない。

Colabでの例: python -m scripts.build_judgment_mix --out data/judgment_mix.jsonl
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from data.convert import jcola_noul, jnli_noul, jsick_score, jsts_score
from data.convert.schema import JevExample
from data.postprocess.rebalance import subsample_by_source
from data.postprocess.shuffle_candidates import shuffle_all

REPO_ID = "fukayatti0/jev-japanese-judgment"


def balance_binary(examples: list[JevExample], seed: int = 0) -> list[JevExample]:
    """答えの文字列(はい/いいえ)ごとに、少ないほうの件数にそろえる。"""
    rng = random.Random(seed)
    by_answer: dict[str, list[JevExample]] = defaultdict(list)
    for ex in examples:
        by_answer[ex.candidates[ex.label]].append(ex)
    n = min(len(v) for v in by_answer.values())
    return [ex for v in by_answer.values() for ex in rng.sample(v, n)]


def cap_per_level(examples: list[JevExample], cap: int, seed: int = 0) -> list[JevExample]:
    rng = random.Random(seed)
    by_level: dict[int, list[JevExample]] = defaultdict(list)
    for ex in examples:
        by_level[ex.label].append(ex)
    return [ex for v in by_level.values() for ex in (rng.sample(v, cap) if len(v) > cap else v)]


def build_mix(
    replay_examples: list[JevExample],
    jsts_cap: int = 1500,
    jsick_cap: int = 800,
    replay_per_source: int = 3400,
    seed: int = 0,
) -> list[JevExample]:
    noul = balance_binary(jnli_noul.convert("train"), seed) + balance_binary(jcola_noul.convert("in_domain_train"), seed)
    score = cap_per_level(jsts_score.convert("train"), jsts_cap, seed) + cap_per_level(
        jsick_score.convert("train"), jsick_cap, seed
    )
    replay = subsample_by_source(replay_examples, max_per_source=replay_per_source, seed=seed)
    mix = shuffle_all(noul + score, seed=seed) + replay  # replayは公開済みデータで既にシャッフル済み
    random.Random(seed).shuffle(mix)
    print(f"noul={len(noul)} score={len(score)} replay={len(replay)} total={len(mix)}")
    return mix


def save(examples: list[JevExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ex.model_dump_json() for ex in examples) + "\n", encoding="utf-8")


def main() -> None:
    from scripts.push_to_hub import pull

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/judgment_mix.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    mix = build_mix(pull(repo_id=REPO_ID), seed=a.seed)
    save(mix, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
