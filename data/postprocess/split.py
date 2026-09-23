"""学習用とホールドアウト評価用にexamplesを分割する。"""

import random

from data.convert.schema import JevExample


def train_val_split(
    examples: list[JevExample], val_size: int = 500, seed: int = 42
) -> tuple[list[JevExample], list[JevExample]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    val = shuffled[:val_size]
    train = shuffled[val_size:]
    return train, val
