"""公開前処理: candidatesの順序をシャッフルし、labelインデックスを再計算する。"""

import random

from data.convert.schema import JevExample


def shuffle_example(example: JevExample, rng: random.Random) -> JevExample:
    indices = list(range(len(example.candidates)))
    rng.shuffle(indices)
    new_candidates = [example.candidates[i] for i in indices]
    new_label = indices.index(example.label)
    return example.model_copy(update={"candidates": new_candidates, "label": new_label})


def shuffle_all(examples: list[JevExample], seed: int = 42) -> list[JevExample]:
    rng = random.Random(seed)
    return [shuffle_example(ex, rng) for ex in examples]
