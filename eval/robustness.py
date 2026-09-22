"""頑健性テスト: 候補順序シャッフルへの一貫性、未知の候補数Kへの汎化。"""

import random

from data.convert.schema import JevExample


def shuffle_order_consistency(
    model_predict_fn, examples: list[JevExample], n_trials: int = 5, seed: int = 0
) -> float:
    """同じ問題を候補順序だけ変えてn_trials回予測させ、
    正解の候補テキスト(順序に依らない)が一致し続ける割合を返す。
    model_predict_fn(examples) -> list[int] (予測label index) を期待する。
    """
    rng = random.Random(seed)
    consistent_count = 0
    for ex in examples:
        answer_text = ex.candidates[ex.label]
        predicted_texts = []
        for _ in range(n_trials):
            indices = list(range(len(ex.candidates)))
            rng.shuffle(indices)
            shuffled = ex.model_copy(
                update={"candidates": [ex.candidates[i] for i in indices]}
            )
            pred_idx = model_predict_fn([shuffled])[0]
            predicted_texts.append(shuffled.candidates[pred_idx])
        if len(set(predicted_texts)) == 1:
            consistent_count += 1
    return consistent_count / len(examples)


def unseen_k_generalization(
    model_predict_fn, examples_by_k: dict[int, list[JevExample]]
) -> dict[int, float]:
    """候補数Kごとのaccuracyを比較し、学習時に見なかったKでの汎化を確認する。"""
    results = {}
    for k, examples in examples_by_k.items():
        preds = model_predict_fn(examples)
        labels = [ex.label for ex in examples]
        correct = sum(p == l for p, l in zip(preds, labels))
        results[k] = correct / len(examples)
    return results
